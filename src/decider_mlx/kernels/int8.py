"""INT8 x u8 MPP matrix product for the Flash affine8 weights (row-quantized activations, integer accumulation).

The activation quantizer and the cooperative INT8 matmul with its scale epilogue are adapted from Shinyoung Kang's
MIT-licensed mxfp4-int8-m5 (e2e/sc_kernels.py); see NOTICE. The weight codes are the unsigned 8-bit codes of the
standard MLX affine8 QMM tensors (no second copy); the zero point is folded in with per-row sums. Rows are
independent (integer accumulation, per-row scales), so a row's output does not depend on the row count.
"""
import mlx.core as mx

HEADER='#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n'

ACT=r'''
const uint row=threadgroup_position_in_grid.x, lid=thread_position_in_threadgroup.x;
const uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
threadgroup float red[8];
threadgroup float factor;
if(row>=M){for(uint k=lid;k<K;k+=256)xq[row*K+k]=0;if(lid==0)xs[row]=0;return;}
float amax=0;
for(uint k=lid;k<K;k+=256)amax=max(amax,abs(float(x[row*K+k])));
amax=simd_max(amax);
if(lane==0)red[sg]=amax;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0){float a=simd_max(lane<8?red[lane]:0.0f);if(lane==0){factor=a>0?127.0f/a:0.0f;xs[row]=a/127.0f;}}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint k=lid;k<K;k+=256)xq[row*K+k]=char(clamp(rint(float(x[row*K+k])*factor),-127.0f,127.0f));
'''

MM=r'''
const uint num_m=threadgroups_per_grid.y,num_n=threadgroups_per_grid.x;
const uint pid=threadgroup_position_in_grid.y*num_n+threadgroup_position_in_grid.x;
const uint per_group=GM*num_n,first_m=(pid/per_group)*GM;
const uint gsz=min(num_m-first_m,uint(GM)),r=pid%per_group;
const int m0=(first_m+r%gsz)*BM,n0=(r/gsz)*BN;
tensor<device int8_t,dextents<int32_t,2>,tensor_inline> tA((device int8_t*)xq,dextents<int32_t,2>(K,M));
tensor<device int8_t,dextents<int32_t,2>,tensor_inline> tB((device int8_t*)wq,dextents<int32_t,2>(K,N));
constexpr auto desc=matmul2d_descriptor(BM,BN,dynamic_length_v<int>,false,true,false,matmul2d_descriptor::mode::multiply);
matmul2d<desc,execution_simdgroups<SG>> op;
auto mA=tA.slice(0,m0);auto mB=tB.slice(0,n0);
auto c=op.template get_destination_cooperative_tensor<decltype(mA),decltype(mB),int32_t>();
op.run(mA,mB,c);
for(uint16_t i=0;i<c.get_capacity();i+=4){
 if(!c.is_valid_element(i))continue;
 auto idx=c.get_multidimensional_index(i);const uint n=n0+idx[0],m=m0+idx[1];
 const float4 a=float4(c[i],c[i+1],c[i+2],c[i+3])*xs[m]*(*(device const float4*)(ws+n));
 for(int j=0;j<4;++j)out[size_t(m)*N+n+j]=static_cast<T>(a[j]);
}
'''


# Same row-quantized signed activation as the existing implementation, plus its sum.
act_source=ACT.replace('if(lid==0)xs[row]=0;', 'if(lid==0){xs[row]=0;sums[row]=0;}')
act_source=act_source.replace('for(uint k=lid;k<K;k+=256)xq[row*K+k]=char(clamp(rint(float(x[row*K+k])*factor),-127.0f,127.0f));',
    '''int total=0;
for(uint k=lid;k<K;k+=256){char q=char(clamp(rint(float(x[row*K+k])*factor),-127.0f,127.0f));xq[row*K+k]=uchar(int(q)+128);total+=int(q);}
total=simd_sum(total);if(lane==0)red[sg]=float(total);threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0){int t=simd_sum(lane<8?int(red[lane]):0);if(lane==0)sums[row]=t;}''')
mixed_source=MM.replace('tensor<device int8_t', 'tensor<device uint8_t').replace('(device int8_t*)','(device uint8_t*)')
mixed_source=mixed_source.replace('float4(c[i],c[i+1],c[i+2],c[i+3])*xs[m]',
    'float4(int4(c[i],c[i+1],c[i+2],c[i+3])-int4(128*sums[m]+16384*K)-128*(*(device const int4*)(weight_sums+n)))*xs[m]')
# A-002: the row count is read at run time (x_shape / xq_shape) instead of a template constant,
# so one pipeline per (N,K) serves every token count. Arithmetic is unchanged.
act_source=act_source.replace('if(row>=M)','if(row>=uint(x_shape[0]))')
mixed_source=mixed_source.replace('dextents<int32_t,2>(K,M)','dextents<int32_t,2>(K,int(xq_shape[0]))')
act_sum=mx.fast.metal_kernel(name='decider_shared_affine8_act_dyn',input_names=['x'],output_names=['xq','xs','sums'],source=act_source)
gemm_mixed=mx.fast.metal_kernel(name='decider_shared_affine8_u8_dyn',input_names=['xq','wq','xs','ws','sums','weight_sums'],output_names=['out'],source=mixed_source,header=HEADER)

BLOCK=32768  # rows per activation-quantize + GEMM call; rows are independent, so the block size never changes the values

def linear(x,codes,row_scale,weight_sums):
    n,k4=codes.shape;k=k4*4;shape=x.shape;x=x.reshape(-1,k);out=[]
    for start in range(0,x.shape[0],BLOCK):
        v=x[start:start+BLOCK];m=v.shape[0];mp=(m+127)//128*128
        xq,xs,sums=act_sum(inputs=[v],template=[('K',k)],grid=(mp*256,1,1),threadgroup=(256,1,1),output_shapes=[(mp,k),(mp,),(mp,)],output_dtypes=[mx.uint8,mx.float32,mx.int32])
        y=gemm_mixed(inputs=[xq,codes,xs,row_scale,sums,weight_sums],template=[('N',n),('K',k),('BM',128),('BN',64),('GM',8),('SG',4),('T',v.dtype)],grid=(n//64*128,mp//128,1),threadgroup=(128,1,1),output_shapes=[(mp,n)],output_dtypes=[v.dtype])[0]
        out.append(y[:m])
    return mx.concatenate(out).reshape(*shape[:-1],n)
