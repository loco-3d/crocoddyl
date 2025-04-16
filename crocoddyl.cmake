# Define ANSY font
string(ASCII 27 Esc)
set(Reset "${Esc}[0m")
set(Bold "${Esc}[1m")

# Function to sanitize the scalar type for filenames
function(crocoddyl_sanitize_scalar_name scalar sanitized_name)
  string(REPLACE "::" "_" sanitized "${scalar}")
  string(REPLACE "<" "_" sanitized "${sanitized}")
  string(REPLACE ">" "" sanitized "${sanitized}")
  string(REPLACE "," "_" sanitized "${sanitized}")
  set(${sanitized_name}
      "${sanitized}"
      PARENT_SCOPE)
endfunction()

# Function to generate cpp files
function(crocoddyl_generate_cpp_files target_lib cpp_format source_dir)
  set(scalar_types ${ARGN})
  foreach(format ${cpp_format})
    message(
      STATUS
        "Generating Crocoddyl's CPP files with ${format} format for ${scalar_types} in ${source_dir}"
    )
    # Add your logic here (e.g., configure_file, file(GLOB), etc.)
  endforeach()
  # Find all .cpp_format.in template files recursively
  file(GLOB_RECURSE ETI_TEMPLATE_FILES CONFIGURE_DEPENDS
       ${CMAKE_SOURCE_DIR}/${source_dir}/**/*.${cpp_format}.in)
  # Remove action codegen if not supported code generation.
  if(NOT BUILD_WITH_CODEGEN_SUPPORT)
    file(GLOB_RECURSE EXCLUDE_TEMPLATES CONFIGURE_DEPENDS
         ${CMAKE_SOURCE_DIR}/${source_dir}/**/*codegen*/*.${cpp_format}.in
         ${CMAKE_SOURCE_DIR}/${source_dir}/**/**/*codegen*/*.${cpp_format}.in)
    foreach(exclude_file ${EXCLUDE_TEMPLATES})
      list(REMOVE_ITEM ETI_TEMPLATE_FILES ${exclude_file})
      list(FIND ETI_TEMPLATE_FILES ${exclude_file} index)
    endforeach()
  endif()
  set(generated_eti_files "")
  foreach(template_file ${ETI_TEMPLATE_FILES})
    # Extract the relative path from ${source_dir}/
    file(RELATIVE_PATH rel_path "${CMAKE_SOURCE_DIR}/${source_dir}"
         "${template_file}")
    # Remove .in extension and get the subdirectory
    string(REPLACE ".${cpp_format}.in" ".cpp" rel_output_file "${rel_path}")
    # Generate full output path in the build directory
    set(output_file "${CMAKE_BINARY_DIR}/${source_dir}/${rel_output_file}")
    # Ensure the output subdirectory exists
    get_filename_component(output_dir "${output_file}" DIRECTORY)
    file(MAKE_DIRECTORY ${output_dir})
    # Generate .cpp file from template for each scalar type
    foreach(scalar ${scalar_types})
      # Sanitize scalar name for filenames
      crocoddyl_sanitize_scalar_name("${scalar}" scalar_filename)
      string(REPLACE ".cpp" "-${scalar_filename}.cpp" scalar_output_file
                     "${output_file}")
      set(SCALAR_TYPE ${scalar})
      configure_file(${template_file} ${scalar_output_file} @ONLY)
      list(APPEND generated_eti_files ${scalar_output_file})
    endforeach()
  endforeach()
  # Add generated files to the target
  target_sources(${target_lib} PRIVATE ${generated_eti_files})
endfunction()
