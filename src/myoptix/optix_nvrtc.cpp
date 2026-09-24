#include "myoptix/optix_nvrtc.h"
#include "myoptix/optix_check.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <nvrtc.h>
#include <fmt/format.h>

#define ABSOLUTE_INCLUDE_DIRS \
  LUXION_OPTIX_INCLUDE, \
  LUXION_EXTERNAL_INCLUDE, \
  LUXION_CUDA_INCLUDE

// OPTIX CUDA RUNTIME COMPILATION ------------------------------

bool readSourceFile( std::string& str, const std::string& filename )
{
    std::ifstream file( filename.c_str(), std::ios::binary );
    if( file.good() )
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>( std::istreambuf_iterator<char>( file ), {} );
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}


void getCuStringFromFile( std::string& cu, const char* filename )
{
    std::vector<std::string> source_locations;

    const std::string file_dir = (std::filesystem::current_path() / "src" / "optixshaders" / filename).string();

    if( readSourceFile( cu, file_dir ) )
    {
        return;
    }

    throw std::runtime_error( "Couldn't open source file " + std::string( file_dir ) );
}

std::string g_nvrtcLog;

void getInputFromCuString( std::string&                    input,
                                  const char*                     cu_source,
                                  const char*                     name,
                                  const char**                    log_string,
                                  const std::vector<const char*>& compiler_options )
{
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR( nvrtcCreateProgram( &prog, cu_source, name, 0, NULL, NULL ) );

    // Gather NVRTC options
    std::vector<const char*> options;

    const std::string base_dir = std::filesystem::current_path().string();

    // // Set sample dir as the primary include path
    // std::string sample_dir;
    // if( sample_directory )   
    // {
    //     sample_dir = std::string( "-I" ) + base_dir + '/' + sample_directory;
    //     options.push_back( sample_dir.c_str() );
    // }

    // Collect include dirs
    std::vector<std::string> include_dirs;
    const char*              abs_dirs[] = {ABSOLUTE_INCLUDE_DIRS};
    const char*              rel_dirs[] = {"src/optixshaders/params"}; // this should be replaced eventually

    for( const char* dir : abs_dirs )
    {
        include_dirs.push_back( std::string( "-I" ) + dir );
    }
    for( const char* dir : rel_dirs )
    {
        include_dirs.push_back( "-I" + base_dir + '/' + dir );
    }
    for( const std::string& dir : include_dirs)
    {
        options.push_back( dir.c_str() );
    }

    // bool optixir = fileExtensionForLoading() == ".optixir";
    bool optixir = true;
    if( optixir )
        options.push_back( "--optix-ir" );
    
    // options.push_back("--std=c++17");
    // options.push_back("--use_fast_math");

    // Collect NVRTC options
    std::copy( std::begin( compiler_options ), std::end( compiler_options ), std::back_inserter( options ) );

    // JIT compile CU to OPTIXIR/PTX
    const nvrtcResult compileRes = nvrtcCompileProgram( prog, (int)options.size(), options.data() );

    // Retrieve log output
    size_t log_size = 0;
    NVRTC_CHECK_ERROR( nvrtcGetProgramLogSize( prog, &log_size ) );
    g_nvrtcLog.resize( log_size );
    if( log_size > 1 )
    {
        NVRTC_CHECK_ERROR( nvrtcGetProgramLog( prog, &g_nvrtcLog[0] ) );
        if( log_string )
            *log_string = g_nvrtcLog.c_str();
    }
    if( compileRes != NVRTC_SUCCESS )
        throw std::runtime_error( "NVRTC Compilation failed.\n" + g_nvrtcLog );

    // Retrieve OPTIXIR/PTX code
    size_t input_size = 0;
    if( optixir )
    {
#if CUDA_VERSION >= 12000
        NVRTC_CHECK_ERROR( nvrtcGetOptiXIRSize( prog, &input_size ) );
        input.resize( input_size );
        NVRTC_CHECK_ERROR( nvrtcGetOptiXIR( prog, &input[0] ) );
#else
        throw std::runtime_error( "OptiX IR support for NVRTC is only available with CUDA 12.0+" );
#endif
    }
    else
    {
        NVRTC_CHECK_ERROR( nvrtcGetPTXSize( prog, &input_size ) );
        input.resize( input_size );
        NVRTC_CHECK_ERROR( nvrtcGetPTX( prog, &input[0] ) );
    }

    // Cleanup
    NVRTC_CHECK_ERROR( nvrtcDestroyProgram( &prog ) );
}
