// CONFIGURATION
#define STREAM_COMPACTION 1
#define MATERIAL_SORTING 0  // enable this if you have a high number of materials
#define RAY_SORTING 1       // morton-sort rays for coherence
#define PROFILE 0           // per-stage cuda event timing

#define RUSSIAN_ROULETTE_MIN_DEPTH 3

// Set this to -1 when profiling off
#define MAX_ITERATIONS -1

// Bump the shader version to recompile shaders. We need a better solution for this
#define SHADER_VER 2.8

#define DRAW_BVH 0

#define OPTIX 1

#define ENABLE_BOX_INTERSECTION     1
#define ENABLE_SPHERE_INTERSECTION  0
#define ENABLE_MESH_INTERSECTION    1

#define UBER_SHADER 1

// deprecated
#define LOAD_FROM_JSON 0