"""W4A8 s4: int8 x int4b (two's complement nibbles) MPP GEMM with the row count M read at run time from meta=[M, MP]
(adapted from the same MIT-licensed INT8 kernels as int8.py; see NOTICE). Tiles: tiles.py.
Rows are independent: per-row activation scale, integer accumulation, so a row's output does not depend on M.
"""
import mlx.core as mx
from .tiles import tile

HEADER = '#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n'
ACT_S4 = 'const uint M=meta[0];\n\nconst uint row=threadgroup_position_in_grid.x, lid=thread_position_in_threadgroup.x;\nconst uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;\nthreadgroup float red[8];\nthreadgroup float factor;\nif(row>=M){for(uint k=lid;k<K;k+=256)xq[row*K+k]=0;if(lid==0)xs[row]=0;return;}\nfloat amax=0;\nfor(uint k=lid;k<K;k+=256)amax=max(amax,abs(float(x[row*K+k])));\namax=simd_max(amax);\nif(lane==0)red[sg]=amax;\nthreadgroup_barrier(mem_flags::mem_threadgroup);\nif(sg==0){float a=simd_max(lane<8?red[lane]:0.0f);if(lane==0){factor=a>0?127.0f/a:0.0f;xs[row]=a/127.0f;}}\nthreadgroup_barrier(mem_flags::mem_threadgroup);\nfor(uint k=lid;k<K;k+=256)xq[row*K+k]=char(clamp(rint(float(x[row*K+k])*factor),-127.0f,127.0f));\n'
GEMM_S4 = '\nconst int M=int(meta[1]);\nconst uint num_m=threadgroups_per_grid.y,num_n=threadgroups_per_grid.x;\nconst uint pid=threadgroup_position_in_grid.y*num_n+threadgroup_position_in_grid.x;\nconst uint per_group=GM*num_n,first_m=(pid/per_group)*GM;\nconst uint gsz=min(num_m-first_m,uint(GM)),r=pid%per_group;\nconst int m0=(first_m+r%gsz)*BM,n0=(r/gsz)*BN;\ntensor<device int8_t,dextents<int32_t,2>,tensor_inline> tA((device int8_t*)xq,dextents<int32_t,2>(K,M));\ntensor<device int4b_format,dextents<int32_t,2>,tensor_inline> tB((device uchar*)codes,dextents<int32_t,2>(K,N));\nconstexpr auto desc=matmul2d_descriptor(BM,BN,dynamic_length_v<int>,false,true,false,matmul2d_descriptor::mode::multiply);\nmatmul2d<desc,execution_simdgroups<SG>> op;\nauto mA=tA.slice(0,m0);auto mB=tB.slice(0,n0);\nauto c=op.template get_destination_cooperative_tensor<decltype(mA),decltype(mB),int32_t>();\nop.run(mA,mB,c);\nfor(uint16_t i=0;i<c.get_capacity();i+=4){\n if(!c.is_valid_element(i))continue;\n auto idx=c.get_multidimensional_index(i);const uint n=n0+idx[0],m=m0+idx[1];\n const float4 a=float4(c[i],c[i+1],c[i+2],c[i+3])*xs[m]*(*(device const float4*)(ws+n));\n for(int j=0;j<4;++j)out[size_t(m)*N+n+j]=static_cast<T>(a[j]);\n}\n'
act_s4 = mx.fast.metal_kernel(name='w4a8_dyn_act_s8', input_names=['x', 'meta'], output_names=['xq', 'xs'], source=ACT_S4)
gemm_s4 = mx.fast.metal_kernel(name='w4a8_dyn_i8_i4', input_names=['xq', 'codes', 'xs', 'ws', 'meta'], output_names=['out'],
                               source=GEMM_S4, header=HEADER)


def linear(x, codes_s4, ws):
    n = codes_s4.shape[0]; k = codes_s4.shape[1] * 8; shape = x.shape; v = x.reshape(-1, k); m = v.shape[0]
    bm, bn, gm, sg = tile(n, k, m)
    if n % bn or k % 32: raise ValueError('need N%BN=0, K%32=0')
    mp = (m + max(bm, 128) - 1) // max(bm, 128) * max(bm, 128)
    meta = mx.array([m, mp], dtype=mx.uint32)
    xq, xs = act_s4(inputs=[v, meta], template=[('K', k)], grid=(mp * 256, 1, 1), threadgroup=(256, 1, 1),
                    output_shapes=[(mp, k), (mp,)], output_dtypes=[mx.int8, mx.float32])
    y = gemm_s4(inputs=[xq, codes_s4, xs, ws, meta],
                template=[('N', n), ('K', k), ('BM', bm), ('BN', bn), ('GM', gm), ('SG', sg), ('T', v.dtype)],
                grid=(n // bn * 32 * sg, mp // bm, 1), threadgroup=(32 * sg, 1, 1), output_shapes=[(mp, n)], output_dtypes=[v.dtype])[0]
    return y[:m].reshape(*shape[:-1], n)
