cmake_minimum_required (VERSION 3.0)
cmake_policy(VERSION 3.0)
set (CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")
#include it to make sure we can find package
include ($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

project(LearnOpenGLPractise)

set(CMAKE_CXX_STANDARD 17) # this does nothing for MSVC, use target_compile_options below
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

set(CHAPTERS
    1.getting_started
    # 2.lighting
    # 3.model_loading
    # 4.advanced_opengl
    # 5.advanced_lighting
    # 6.pbr
    # 7.in_practice
)

set(1.getting_started
    1.1creating_a_window
    2.1.hello_triangle
    2.2.hello_triangle_indexed
    3.1.shaders_uniform
)

# set(2.lighting
#     1.colors
#     2.1.basic_lighting_diffuse
#     2.2.basic_lighting_specular
#     3.1.materials
#     3.2.materials_exercise1
#     4.1.lighting_maps_diffuse_map
#     4.2.lighting_maps_specular_map
#     4.4.lighting_maps_exercise4
#     5.1.light_casters_directional
#     5.2.light_casters_point
#     5.3.light_casters_spot
#     5.4.light_casters_spot_soft
#     6.multiple_lights
# )

# set(3.model_loading
#     1.model_loading
# )

# set(4.advanced_opengl
#     1.1.depth_testing
#     1.2.depth_testing_view
#     2.stencil_testing
#     3.1.blending_discard
#     3.2.blending_sort
#     5.1.framebuffers
#     5.2.framebuffers_exercise1
#     6.1.cubemaps_skybox
#     6.2.cubemaps_environment_mapping
#     8.advanced_glsl_ubo
#     9.1.geometry_shader_houses
#     9.2.geometry_shader_exploding
#     9.3.geometry_shader_normals
#     10.1.instancing_quads
#     10.2.asteroids
#     10.3.asteroids_instanced
#     11.1.anti_aliasing_msaa
#     11.2.anti_aliasing_offscreen
# )

# set(5.advanced_lighting
#     1.advanced_lighting
#     2.gamma_correction
#     3.1.1.shadow_mapping_depth
#     3.1.2.shadow_mapping_base
#     3.1.3.shadow_mapping
#     3.2.1.point_shadows
#     3.2.2.point_shadows_soft
#     4.normal_mapping
#     5.1.parallax_mapping
#     5.2.steep_parallax_mapping
#     5.3.parallax_occlusion_mapping
#     6.hdr
#     7.bloom
#     8.1.deferred_shading
#     8.2.deferred_shading_volumes
#     9.ssao
# )

# set(6.pbr
#     1.1.lighting
#     1.2.lighting_textured
#     2.1.1.ibl_irradiance_conversion
#     2.1.2.ibl_irradiance
#     2.2.1.ibl_specular
#     2.2.2.ibl_specular_textured
# )

# set(7.in_practice
#     1.debugging
#     2.text_rendering
#     #3.2d_game
# )

# set(GUEST_ARTICLES
# 	8.guest/2020/oit
# 	8.guest/2020/skeletal_animation
# 	8.guest/2021/1.scene/1.scene_graph
# 	8.guest/2021/1.scene/2.frustum_culling
# 	8.guest/2021/2.csm
# 	8.guest/2021/3.tessellation/terrain_gpu_dist
# 	8.guest/2021/3.tessellation/terrain_cpu_src
# 	8.guest/2021/4.dsa
# 	8.guest/2022/5.computeshader_helloworld
# 	8.guest/2022/6.physically_based_bloom
# 	8.guest/2022/7.area_lights/1.area_light
# 	8.guest/2022/7.area_lights/2.multiple_area_lights
# )


set(LIBS glfw glad::glad)
# first create relevant static libraries required for other projects
#add_library(STB_IMAGE "src/stb_image.cpp")
#set(LIBS ${LIBS} STB_IMAGE)


macro(makeLink src dest target)
  add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink ${src} ${dest}  DEPENDS  ${dest} COMMENT "mklink ${src} -> ${dest}")
endmacro()

function(create_project_from_sources chapter demo)
	file(GLOB SOURCE
            "src/${chapter}/${demo}/*.h"
            "src/${chapter}/${demo}/*.cpp"
            "src/${chapter}/${demo}/*.vs"
            "src/${chapter}/${demo}/*.fs"
            "src/${chapter}/${demo}/*.tcs"
            "src/${chapter}/${demo}/*.tes"
            "src/${chapter}/${demo}/*.gs"
            "src/${chapter}/${demo}/*.cs"
    )
	if (demo STREQUAL "")
		SET(replaced "")
		string(REPLACE "/" "_" replaced ${chapter})
		set(NAME ${replaced})
	else()
		set(NAME "${chapter}__${demo}")
	endif()
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} ${LIBS})
    if(MSVC)
		target_compile_options(${NAME} PRIVATE /std:c++17 /MP)
        target_link_options(${NAME} PUBLIC /ignore:4099)
    endif(MSVC)
    if(WIN32)
        set_target_properties(${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/${chapter}")
        set_target_properties(${NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/${chapter}/Debug")
    elseif(UNIX AND NOT APPLE)
        set_target_properties(${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/${chapter}")
    elseif(APPLE)
        set_target_properties(${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/${chapter}")
        set_target_properties(${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin/${chapter}")
        set_target_properties(${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin/${chapter}")
    endif(WIN32)
    # copy shader files to build directory
    file(GLOB SHADERS
             "src/${chapter}/${demo}/*.vs"
             # "src/${chapter}/${demo}/*.frag"
             "src/${chapter}/${demo}/*.fs"
             "src/${chapter}/${demo}/*.tcs"
             "src/${chapter}/${demo}/*.tes"
             "src/${chapter}/${demo}/*.gs"
             "src/${chapter}/${demo}/*.cs"
    )
	# copy dlls
	file(GLOB DLLS "dlls/*.dll")
    foreach(SHADER ${SHADERS})
        if(WIN32)
            # configure_file(${SHADER} "test")
            add_custom_command(TARGET ${NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${SHADER} $<TARGET_FILE_DIR:${NAME}>)
			add_custom_command(TARGET ${NAME} PRE_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${DLLS} $<TARGET_FILE_DIR:${NAME}>)
        elseif(UNIX AND NOT APPLE)
            file(COPY ${SHADER} DESTINATION ${CMAKE_SOURCE_DIR}/bin/${chapter})
        elseif(APPLE)
            # create symbolic link for *.vs *.fs *.gs
            get_filename_component(SHADERNAME ${SHADER} NAME)
            makeLink(${SHADER} ${CMAKE_SOURCE_DIR}/bin/${chapter}/${SHADERNAME} ${NAME})
        endif(WIN32)
    endforeach(SHADER)
    # if compiling for visual studio, also use configure file for each project (specifically to set up working directory)
    # if(MSVC)
    #     configure_file(${CMAKE_SOURCE_DIR}/configuration/visualstudio.vcxproj.user.in ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.vcxproj.user @ONLY)
    # endif(MSVC)
endfunction()

# then create a project file per tutorial
foreach(CHAPTER ${CHAPTERS})
    foreach(DEMO ${${CHAPTER}})
		create_project_from_sources(${CHAPTER} ${DEMO})
    endforeach(DEMO)
endforeach(CHAPTER)

include_directories(${CMAKE_SOURCE_DIR}/includes)
