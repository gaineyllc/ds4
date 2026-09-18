/* Build-time syntax check for metal/engram.metal.
 *
 * ds4 compiles its shaders at runtime with newLibraryWithSource, and the Metal
 * toolchain is not installed on this machine, so this harness runs the same
 * runtime compiler over one file and confirms the kernels exist and can build
 * a pipeline. It does not run them.
 *
 * clang -fobjc-arc -framework Foundation -framework Metal \
 *     -o /tmp/engram_msl_check tools/engram_msl_check.m
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, const char **argv) {
    @autoreleasepool {
        const char *path = (argc > 1) ? argv[1] : "metal/engram.metal";
        NSError *error = nil;
        NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path]
                                                  encoding:NSUTF8StringEncoding
                                                     error:&error];
        if (!src) {
            fprintf(stderr, "read failed: %s\n", [[error localizedDescription] UTF8String]);
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "no Metal device\n"); return 2; }

        MTLCompileOptions *options = [MTLCompileOptions new];
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:options error:&error];
        if (!lib) {
            fprintf(stderr, "MSL compile failed:\n%s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }
        const char *want[] = {
            "kernel_engram_fused_gather_gate",
            "kernel_engram_fused_gather_gate_store",
            "kernel_engram_gather_rows",
        };
        int rc = 0;
        for (size_t i = 0; i < sizeof(want) / sizeof(want[0]); i++) {
            NSString *name = [NSString stringWithUTF8String:want[i]];
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) { fprintf(stderr, "missing function: %s\n", want[i]); rc = 1; continue; }
            NSError *pe = nil;
            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&pe];
            if (!pso) {
                fprintf(stderr, "pipeline failed for %s: %s\n",
                        want[i], [[pe localizedDescription] UTF8String]);
                rc = 1;
                continue;
            }
            printf("ok %-42s maxTotalThreadsPerThreadgroup=%lu threadExecutionWidth=%lu\n",
                   want[i],
                   (unsigned long)[pso maxTotalThreadsPerThreadgroup],
                   (unsigned long)[pso threadExecutionWidth]);
        }
        return rc;
    }
}
