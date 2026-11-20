cmake_minimum_required(VERSION 3.28)
project(Phanerozoic LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PHX_ENABLE_TSAN "Build with ThreadSanitizer" OFF)
option(PHX_BUILD_TESTS "Build unit tests" ON)

if (PHX_ENABLE_TSAN)
    add_compile_options(-fsanitize=thread -g)
    add_link_options(-fsanitize=thread)
endif ()

# ---------------------------------------------------------------- dependencies
include(FetchContent)

find_package(Vulkan REQUIRED)            # headers (+ loader, volk loads dynamically anyway)
find_package(Threads REQUIRED)
find_package(glm REQUIRED)

find_package(glfw3 3.4 QUIET)
if (NOT glfw3_FOUND)
    FetchContent_Declare(glfw
        URL https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
endif ()

FetchContent_Declare(volk
    URL https://github.com/zeux/volk/archive/refs/tags/1.4.304.tar.gz)
FetchContent_Declare(vma   # header-only: SOURCE_SUBDIR keeps its CMake project out of our build
    URL https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/archive/refs/tags/v3.3.0.tar.gz
    SOURCE_SUBDIR _none)
set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
FetchContent_Declare(cgltf URL https://github.com/jkuhlmann/cgltf/archive/refs/tags/v1.14.tar.gz SOURCE_SUBDIR _none)
FetchContent_Declare(meshoptimizer URL https://github.com/zeux/meshoptimizer/archive/refs/tags/v0.22.tar.gz)
FetchContent_Declare(stb URL https://github.com/nothings/stb/archive/f58f558c120e9b32c217290b80bad1a0729fbb2c.tar.gz SOURCE_SUBDIR _none)
FetchContent_MakeAvailable(volk vma cgltf meshoptimizer stb)

# Jolt Physics (MIT): player controller and static colliders around the player (M6).
FetchContent_Declare(JoltPhysics
    URL https://github.com/jrouwe/JoltPhysics/archive/refs/tags/v5.2.0.tar.gz
    SOURCE_SUBDIR Build)
set(TARGET_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(TARGET_HELLO_WORLD OFF CACHE BOOL "" FORCE)
set(TARGET_PERFORMANCE_TEST OFF CACHE BOOL "" FORCE)
set(TARGET_SAMPLES OFF CACHE BOOL "" FORCE)
set(TARGET_VIEWER OFF CACHE BOOL "" FORCE)
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)
set(PROFILER_IN_DEBUG_AND_RELEASE OFF CACHE BOOL "" FORCE)
set(CPP_RTTI_ENABLED ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(JoltPhysics)
target_compile_options(Jolt PRIVATE -O2) # the character sweep is far too slow at -O0

# NVIDIA DLSS (NGX SDK, "NVIDIA RTX SDKs" licence: data/README.md). Linux x86_64 only;
# headers, the static NGX loader and the DLSS runtime are fetched at configure time.
option(PHX_DLSS "NVIDIA DLSS Super Resolution / DLAA via NGX" ON)
set(PHX_DLSS_VERSION 310.9.16)
set(PHX_DLSS_AVAILABLE OFF)
if (PHX_DLSS AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(PHX_DLSS_DIR ${CMAKE_BINARY_DIR}/_deps/dlss-sdk-${PHX_DLSS_VERSION})
    set(PHX_DLSS_URL https://raw.githubusercontent.com/NVIDIA/DLSS/v${PHX_DLSS_VERSION})
    set(PHX_DLSS_FILES
        LICENSE.txt
        lib/Linux_x86_64/libnvsdk_ngx.a
        lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.${PHX_DLSS_VERSION})
    foreach (h nvsdk_ngx.h nvsdk_ngx_defs.h nvsdk_ngx_defs_dlssd.h nvsdk_ngx_defs_dlssg.h nvsdk_ngx_defs_vk.h
               nvsdk_ngx_helpers.h nvsdk_ngx_helpers_vk.h nvsdk_ngx_helpers_dlssd.h nvsdk_ngx_helpers_dlssd_vk.h
               nvsdk_ngx_helpers_dlssg.h nvsdk_ngx_helpers_dlssg_vk.h nvsdk_ngx_params.h nvsdk_ngx_params_dlssd.h
               nvsdk_ngx_params_dlssg.h nvsdk_ngx_vk.h)
        list(APPEND PHX_DLSS_FILES include/${h})
    endforeach ()
    set(PHX_DLSS_OK ON)
    foreach (f ${PHX_DLSS_FILES})
        if (NOT EXISTS ${PHX_DLSS_DIR}/${f})
            message(STATUS "DLSS SDK ${PHX_DLSS_VERSION}: downloading ${f}")
            file(DOWNLOAD ${PHX_DLSS_URL}/${f} ${PHX_DLSS_DIR}/${f}.part STATUS st TLS_VERIFY ON)
            list(GET st 0 code)
            if (code EQUAL 0)
                file(RENAME ${PHX_DLSS_DIR}/${f}.part ${PHX_DLSS_DIR}/${f})
            else ()
                file(REMOVE ${PHX_DLSS_DIR}/${f}.part)
                message(WARNING "DLSS SDK download failed (${f}): ${st}; building without DLSS")
                set(PHX_DLSS_OK OFF)
                break()
            endif ()
        endif ()
    endforeach ()
    if (PHX_DLSS_OK)
        add_library(phx_ngx STATIC IMPORTED)
        set_target_properties(phx_ngx PROPERTIES
            IMPORTED_LOCATION ${PHX_DLSS_DIR}/lib/Linux_x86_64/libnvsdk_ngx.a
            INTERFACE_INCLUDE_DIRECTORIES ${PHX_DLSS_DIR}/include
            INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS};stdc++")
        # The DLSS runtime ("snippet") is loaded by NGX from the application directory.
        configure_file(${PHX_DLSS_DIR}/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.${PHX_DLSS_VERSION}
                       ${CMAKE_BINARY_DIR}/libnvidia-ngx-dlss.so.${PHX_DLSS_VERSION} COPYONLY)
        set(PHX_DLSS_AVAILABLE ON)
    endif ()
endif ()
add_library(cgltf INTERFACE)
target_include_directories(cgltf SYSTEM INTERFACE ${cgltf_SOURCE_DIR})
add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE ${stb_SOURCE_DIR})
add_library(vma INTERFACE)
target_include_directories(vma SYSTEM INTERFACE ${vma_SOURCE_DIR}/include)

# ---------------------------------------------------------------- jobs (no GPU deps, testable alone)
add_library(phx_jobs STATIC
    engine/jobs/JobSystem.cpp)
target_include_directories(phx_jobs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(phx_jobs PUBLIC Threads::Threads)
target_compile_options(phx_jobs PRIVATE -Wall -Wextra)

# ---------------------------------------------------------------- engine
add_library(phx_astro STATIC
    engine/astro/Astro.cpp
    engine/astro/StarCatalog.cpp
    engine/core/Log.cpp)
target_include_directories(phx_astro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(phx_astro PUBLIC glm::glm)
target_compile_options(phx_astro PRIVATE -Wall -Wextra)

add_library(phx_engine STATIC
    engine/core/Png.cpp
    engine/gpu/Resources.cpp
    engine/gpu/GpuProfiler.cpp
    engine/gpu/Textures.cpp
    engine/render/SkyRenderer.cpp
    engine/render/TerrainRenderer.cpp
    engine/render/TerrainShadows.cpp
    engine/render/HiZ.cpp
    engine/render/CascadedShadows.cpp
    engine/render/PerfOverlay.cpp
    engine/render/Upscaler.cpp
    engine/world/NaturePlacement.cpp
    engine/world/Climate.cpp
    engine/game/PhysicsWorld.cpp
    engine/game/Player.cpp
    engine/render/VegetationRenderer.cpp
    engine/world/TreeMeshes.cpp
    engine/world/Environment.cpp
    engine/world/HeightField.cpp
    engine/platform/Window.cpp
    engine/gpu/VulkanContext.cpp
    engine/gpu/Swapchain.cpp
    engine/gpu/VmaImpl.cpp
    engine/render/Renderer.cpp
    engine/Engine.cpp)
target_include_directories(phx_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(phx_engine PUBLIC phx_jobs phx_astro volk vma glfw Vulkan::Headers stb Jolt)
target_compile_definitions(phx_engine PRIVATE
    PHX_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
    PHX_DATA_DIR="${CMAKE_BINARY_DIR}/data"
    PHX_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/raw/polyhaven")
target_compile_definitions(phx_engine PUBLIC GLFW_INCLUDE_NONE)
if (PHX_DLSS_AVAILABLE)
    target_link_libraries(phx_engine PUBLIC phx_ngx)
    target_compile_definitions(phx_engine PUBLIC PHX_HAS_DLSS=1 PHX_DLSS_VERSION="${PHX_DLSS_VERSION}")
endif ()
# Vulkan structs are initialised as `T x{sType}`; the remaining members are zeroed by design.
target_compile_options(phx_engine PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
set_source_files_properties(engine/game/PhysicsWorld.cpp engine/world/Climate.cpp PROPERTIES COMPILE_OPTIONS -O2)

# ---------------------------------------------------------------- shaders (SPIR-V, built into build/shaders)
find_program(GLSLC glslc REQUIRED)
file(GLOB PHX_SHADER_INCLUDES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl)
file(GLOB PHX_SHADER_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert
                                              ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag
                                              ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp)
set(PHX_SPV_OUTPUTS)
foreach (src ${PHX_SHADER_SOURCES})
    get_filename_component(name ${src} NAME)
    set(out ${CMAKE_BINARY_DIR}/shaders/${name}.spv)
    add_custom_command(OUTPUT ${out}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${GLSLC} --target-env=vulkan1.3 -O -I ${CMAKE_CURRENT_SOURCE_DIR}/shaders -o ${out} ${src}
        DEPENDS ${src} ${PHX_SHADER_INCLUDES} VERBATIM)
    list(APPEND PHX_SPV_OUTPUTS ${out})
endforeach ()
add_custom_target(phx_shaders DEPENDS ${PHX_SPV_OUTPUTS})

# ---------------------------------------------------------------- data baking
add_executable(phx_bake_stars tools/bake/BakeStars.cpp)
target_link_libraries(phx_bake_stars PRIVATE phx_astro)

set(PHX_HYG_CSV ${CMAKE_CURRENT_SOURCE_DIR}/data/raw/hygdata_v41.csv)
set(PHX_HYG_URL https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv)
add_custom_command(OUTPUT ${PHX_HYG_CSV}
    COMMAND ${CMAKE_COMMAND} -DURL=${PHX_HYG_URL} -DDEST=${PHX_HYG_CSV} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Download.cmake
    COMMENT "Downloading HYG star database (CC BY-SA 4.0)" VERBATIM)
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/data/stars.bin
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
    COMMAND phx_bake_stars ${PHX_HYG_CSV} ${CMAKE_BINARY_DIR}/data/stars.bin 8.5
    DEPENDS phx_bake_stars ${PHX_HYG_CSV} VERBATIM)
add_custom_target(phx_star_data DEPENDS ${CMAKE_BINARY_DIR}/data/stars.bin)

# Global DEM: PALEOMAP PaleoDEM 0 Ma (modern geography; Scotese & Wright 2018, CC BY 4.0).
# Eocene corrections are applied at load time (engine/world/HeightField.cpp).
find_package(HDF5 COMPONENTS C)
set(PHX_DEM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data/raw/paleodem)
set(PHX_DEM_NC ${PHX_DEM_DIR}/Map01_PALEOMAP_1deg_Holocene_0Ma.nc)
set(PHX_DEM_ZIP ${PHX_DEM_DIR}/pd1deg.zip)
set(PHX_DEM_URL https://www.earthbyte.org/webdav/ftp/Data_Collections/Scotese_Wright_2018_PaleoDEM/Scotese_Wright_2018_Maps_1-88_1degX1deg_PaleoDEMS_nc.zip)
if (HDF5_FOUND)
    add_executable(phx_bake_paleodem tools/bake/BakePaleoDem.cpp)
    target_include_directories(phx_bake_paleodem PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(phx_bake_paleodem PRIVATE ${HDF5_C_LIBRARIES})
    add_custom_command(OUTPUT ${PHX_DEM_ZIP}
        COMMAND ${CMAKE_COMMAND} -DURL=${PHX_DEM_URL} -DDEST=${PHX_DEM_ZIP} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Download.cmake
        COMMENT "Downloading PALEOMAP PaleoDEMs (CC BY 4.0)" VERBATIM)
    add_custom_command(OUTPUT ${PHX_DEM_NC}
        COMMAND ${CMAKE_COMMAND} -E tar xf ${PHX_DEM_ZIP} --
            Scotese_Wright_2018_Maps_1-88_1degX1deg_PaleoDEMS_nc_v2/Map01_PALEOMAP_1deg_Holocene_0Ma.nc
        COMMAND ${CMAKE_COMMAND} -E rename
            ${PHX_DEM_DIR}/Scotese_Wright_2018_Maps_1-88_1degX1deg_PaleoDEMS_nc_v2/Map01_PALEOMAP_1deg_Holocene_0Ma.nc ${PHX_DEM_NC}
        WORKING_DIRECTORY ${PHX_DEM_DIR}
        DEPENDS ${PHX_DEM_ZIP} VERBATIM)
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/data/dem_global.bin
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
        COMMAND phx_bake_paleodem ${PHX_DEM_NC} ${CMAKE_BINARY_DIR}/data/dem_global.bin
        DEPENDS phx_bake_paleodem ${PHX_DEM_NC} VERBATIM)
    add_custom_target(phx_dem_data DEPENDS ${CMAKE_BINARY_DIR}/data/dem_global.bin)
else ()
    message(WARNING "HDF5 not found: the global DEM cannot be baked; terrain falls back to procedural continents")
    add_custom_target(phx_dem_data)
endif ()

# Regional 30 m DEM: Copernicus GLO-30 (free licence, attribution required, see
# data/README.md). PHX_DEM30_CELLS lists 1° cells as LATxLON of the south-west corner.
set(PHX_DEM30_CELLS "33x109;33x110;34x109;34x110" CACHE STRING "Copernicus GLO-30 cells to bake")
find_package(TIFF)
set(PHX_DEM30_OUTPUTS)
if (TIFF_FOUND)
    add_executable(phx_bake_dem30 tools/bake/BakeDem30.cpp)
    target_link_libraries(phx_bake_dem30 PRIVATE TIFF::TIFF)
    foreach (cell ${PHX_DEM30_CELLS})
        string(REPLACE "x" ";" parts ${cell})
        list(GET parts 0 lat)
        list(GET parts 1 lon)
        set(ns N)
        set(ew E)
        set(alat ${lat})
        set(alon ${lon})
        if (lat LESS 0)
            set(ns S)
            math(EXPR alat "0 - ${lat}")
        endif ()
        if (lon LESS 0)
            set(ew W)
            math(EXPR alon "0 - ${lon}")
        endif ()
        string(LENGTH "${alat}" l1)
        if (l1 LESS 2)
            set(alat "0${alat}")
        endif ()
        set(alon3 "00${alon}")
        string(LENGTH "${alon3}" l2)
        math(EXPR start "${l2} - 3")
        string(SUBSTRING "${alon3}" ${start} 3 alon3)
        set(name Copernicus_DSM_COG_10_${ns}${alat}_00_${ew}${alon3}_00_DEM)
        set(tif ${CMAKE_CURRENT_SOURCE_DIR}/data/raw/dem30/${name}.tif)
        set(out ${CMAKE_BINARY_DIR}/data/dem30/${ns}${alat}_${ew}${alon3}.phxdem)
        add_custom_command(OUTPUT ${tif}
            COMMAND ${CMAKE_COMMAND} -DURL=https://copernicus-dem-30m.s3.amazonaws.com/${name}/${name}.tif -DDEST=${tif}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Download.cmake
            COMMENT "Downloading Copernicus GLO-30 ${name}" VERBATIM)
        add_custom_command(OUTPUT ${out}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data/dem30
            COMMAND phx_bake_dem30 ${tif} ${out} ${lat} ${lon}
            DEPENDS phx_bake_dem30 ${tif} VERBATIM)
        list(APPEND PHX_DEM30_OUTPUTS ${out})
    endforeach ()
else ()
    message(WARNING "libtiff not found: regional 30 m DEM disabled")
endif ()
add_custom_target(phx_dem30_data DEPENDS ${PHX_DEM30_OUTPUTS})

# Poly Haven assets (CC0): fetched by tools/assets/fetch_polyhaven.py from data/assets/polyhaven.txt.
# Trees are baked into near-field LODs + full meshes (impostors are baked at start-up from those).
find_package(Python3 COMPONENTS Interpreter)
set(PHX_PH_RAW ${CMAKE_CURRENT_SOURCE_DIR}/data/raw/polyhaven)
set(PHX_PH_STAMP ${PHX_PH_RAW}/.fetched)
add_custom_command(OUTPUT ${PHX_PH_STAMP}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/assets/fetch_polyhaven.py
            ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/polyhaven.txt ${PHX_PH_RAW}
    COMMAND ${CMAKE_COMMAND} -E touch ${PHX_PH_STAMP}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data/assets/polyhaven.txt
    COMMENT "Fetching Poly Haven assets (CC0)" VERBATIM)
add_executable(phx_bake_tree tools/bake/BakeTree.cpp)
target_link_libraries(phx_bake_tree PRIVATE cgltf meshoptimizer glm::glm)
set_source_files_properties(tools/bake/BakeTree.cpp PROPERTIES COMPILE_OPTIONS "-O2")
# "species id | asset | height (m) | extra args (comma separated, - for none)"
set(PHX_TREES
    "0|fir_tree_01|26|--crop,-3.6,3.6,--lod0,450000,--lod1,120000,--lod2,20000"
    "1|island_tree_01|9|--lod0,450000,--lod1,120000,--lod2,20000"
    "2|tree_small_02|9|--lod0,450000,--lod1,120000,--lod2,20000"
    "3|fern_02|0.55|--crop,-0.6,0.6,--lod0,4000,--lod1,1500,--lod2,600"
    "4|rock_moss_set_01|1.0|--crop,-1.5,1.5,--lod0,12000,--lod1,4000,--lod2,800"
    "5|boulder_01|2.4|--lod0,30000,--lod1,8000,--lod2,1500"
    "6|grass_medium_01|0.45|--lod0,6000,--lod1,2000,--lod2,600"
    "7|shrub_02|1.4|--lod0,9000,--lod1,3000,--lod2,900"
    "8|searsia_lucida|4.5|--lod0,60000,--lod1,15000,--lod2,3500")
set(PHX_TREE_OUTPUTS)
foreach (t ${PHX_TREES})
    string(REPLACE "|" ";" f "${t}")
    list(GET f 0 sid)
    list(GET f 1 asset)
    list(GET f 2 h)
    list(GET f 3 extra)
    string(REPLACE "," ";" extra "${extra}")
    if (extra STREQUAL "-")
        set(extra "")
    endif ()
    set(out ${CMAKE_BINARY_DIR}/data/trees/species${sid}.phxtree)
    add_custom_command(OUTPUT ${out}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data/trees
        COMMAND phx_bake_tree ${PHX_PH_RAW}/${asset}/${asset}_1k.gltf ${out} ${h} ${extra}
        DEPENDS phx_bake_tree ${PHX_PH_STAMP} VERBATIM)
    list(APPEND PHX_TREE_OUTPUTS ${out})
endforeach ()
add_custom_target(phx_tree_data DEPENDS ${PHX_TREE_OUTPUTS})

# ---------------------------------------------------------------- game executable
add_executable(Phanerozoic game/main.cpp)
target_link_libraries(Phanerozoic PRIVATE phx_engine)
add_dependencies(Phanerozoic phx_shaders phx_star_data phx_dem_data phx_dem30_data phx_tree_data)

# ---------------------------------------------------------------- tests
if (PHX_BUILD_TESTS)
    enable_testing()
    add_executable(phx_job_tests tests/JobSystemTests.cpp)
    target_link_libraries(phx_job_tests PRIVATE phx_jobs)
    add_test(NAME job_system COMMAND phx_job_tests)
    add_executable(phx_astro_tests tests/AstroTests.cpp)
    target_link_libraries(phx_astro_tests PRIVATE phx_astro)
    add_test(NAME astro COMMAND phx_astro_tests)
endif ()

# Third-party implementation TU: don't hold it to our warning level.
set_source_files_properties(engine/gpu/VmaImpl.cpp PROPERTIES COMPILE_OPTIONS "-w")
set_source_files_properties(engine/gpu/Textures.cpp PROPERTIES COMPILE_OPTIONS "-O2;-Wno-unused-function")

# Hot CPU paths (tile generation runs millions of noise evaluations): optimise them even
# in Debug builds so streaming keeps up while the rest stays debuggable.
set_source_files_properties(engine/world/HeightField.cpp engine/render/TerrainRenderer.cpp
    engine/render/VegetationRenderer.cpp engine/render/TerrainShadows.cpp
    PROPERTIES COMPILE_OPTIONS "-O2")
