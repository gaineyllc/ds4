// membw: achievable GPU streaming read bandwidth on Apple silicon.
// Reads a buffer far larger than the SLC, N iterations, reports GB/s.
// Build: clang -fobjc-arc -framework Metal -framework Foundation -O2 membw.m -o membw
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>

static const char *kSrc =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void rd(device const uint4 *in [[buffer(0)]], device uint *out [[buffer(1)]],\n"
"               constant uint &n16_per_thread [[buffer(2)]], constant ulong &n16 [[buffer(3)]],\n"
"               uint tid [[thread_position_in_grid]], uint nth [[threads_per_grid]]) {\n"
"    uint4 acc = 0;\n"
"    // grid-stride: consecutive threads read consecutive 16B -> fully coalesced\n"
"    for (ulong i = tid; i < n16; i += nth) acc ^= in[i];\n"
"    uint v = acc.x ^ acc.y ^ acc.z ^ acc.w;\n"
"    if (v == 0xdeadbeefu) out[0] = v;\n"
"}\n"
"kernel void rdstride(device const uint4 *in [[buffer(0)]], device uint *out [[buffer(1)]],\n"
"               constant uint &chunk16 [[buffer(2)]], constant ulong &n16 [[buffer(3)]],\n"
"               uint tid [[thread_position_in_grid]], uint nth [[threads_per_grid]]) {\n"
"    // each thread streams its own contiguous chunk (matvec-row style)\n"
"    uint4 acc = 0;\n"
"    ulong base = (ulong)tid * chunk16;\n"
"    for (uint i = 0; i < chunk16 && base + i < n16; i++) acc ^= in[base + i];\n"
"    uint v = acc.x ^ acc.y ^ acc.z ^ acc.w;\n"
"    if (v == 0xdeadbeefu) out[0] = v;\n"
"}\n";

static double now_s(void){ static mach_timebase_info_data_t tb; if(!tb.denom) mach_timebase_info(&tb); return mach_absolute_time()*(double)tb.numer/tb.denom/1e9; }

int main(int argc, char **argv) {
    @autoreleasepool {
        size_t gib = argc > 1 ? atoi(argv[1]) : 8;
        int iters = argc > 2 ? atoi(argv[2]) : 10;
        size_t bytes = gib << 30;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s  recommendedMaxWorkingSet=%.1f GiB  hasUnifiedMemory=%d\n",
               dev.name.UTF8String, dev.recommendedMaxWorkingSetSize/1073741824.0, dev.hasUnifiedMemory);
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&err];
        if (!lib) { printf("compile: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rd"] error:&err];
        id<MTLComputePipelineState> pso2 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rdstride"] error:&err];
        id<MTLCommandQueue> q = [dev newCommandQueue];
        // Fill on CPU so pages are resident and non-zero.
        id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) { printf("alloc %zu GiB failed\n", gib); return 1; }
        uint32_t *p = buf.contents; size_t n32 = bytes/4;
        for (size_t i = 0; i < n32; i += 1024) p[i] = (uint32_t)i * 2654435761u;
        id<MTLBuffer> out = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        unsigned long n16 = bytes / 16;
        // total threads for grid-stride: many threadgroups of 256
        NSUInteger tg = 256, ngroups = 4096; unsigned nper = 0;
        printf("buffer %zu GiB, %d iters (first discarded)\n\n", gib, iters);
        for (int mode = 0; mode < 2; mode++) {
            id<MTLComputePipelineState> ps = mode ? pso2 : pso;
            NSUInteger groups = ngroups;
            unsigned chunk16 = 0;
            if (mode) { // strided: 16 KiB per thread
                chunk16 = 1024; groups = (NSUInteger)((n16 + chunk16 - 1) / chunk16 + tg - 1) / tg;
            }
            double best = 0, sum = 0; int cnt = 0;
            for (int it = 0; it < iters; it++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                [enc setBuffer:buf offset:0 atIndex:0];
                [enc setBuffer:out offset:0 atIndex:1];
                unsigned arg2 = mode ? chunk16 : nper;
                [enc setBytes:&arg2 length:4 atIndex:2];
                [enc setBytes:&n16 length:8 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
                [enc endEncoding];
                double t0 = now_s();
                [cb commit]; [cb waitUntilCompleted];
                double t1 = now_s();
                double gpu = cb.GPUEndTime - cb.GPUStartTime;
                double gbs_wall = bytes / (t1 - t0) / 1e9, gbs_gpu = bytes / gpu / 1e9;
                printf("%-8s iter %2d: wall %.1f GB/s  gpu-timestamps %.1f GB/s  (%.1f ms)\n",
                       mode ? "strided" : "coalesced", it, gbs_wall, gbs_gpu, gpu*1e3);
                if (it > 0) { sum += gbs_gpu; cnt++; if (gbs_gpu > best) best = gbs_gpu; }
            }
            printf("=> %-9s best %.1f GB/s, mean %.1f GB/s (gpu timestamps, %d runs)\n\n", mode ? "strided" : "coalesced", best, sum/cnt, cnt);
        }
    }
    return 0;
}
