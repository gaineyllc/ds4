/* bench_deepseek4_vision VISION.gguf IMAGE [MAX_TOKENS|native] [ITERS] [DUMP.bin]
 * Times the DeepSeek V4.1 vision encoder alone (no language model load). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ds4_deepseek4_vision_bench(const char *vision_path, const char *image_path,
                               uint32_t max_tokens, int iters, const char *dump_path);

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s VISION.gguf IMAGE [MAX_TOKENS|native] [ITERS] [DUMP.bin]\n", argv[0]);
        return 2;
    }
    uint32_t budget = 0;
    if (argc > 3) budget = !strcmp(argv[3], "native") ? UINT32_MAX : (uint32_t)strtoul(argv[3], NULL, 10);
    const int iters = argc > 4 ? atoi(argv[4]) : 3;
    return ds4_deepseek4_vision_bench(argv[1], argv[2], budget, iters, argc > 5 ? argv[5] : NULL) ? 0 : 1;
}
