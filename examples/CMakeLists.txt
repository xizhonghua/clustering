include_directories(  
  ${CMAKE_SOURCE_DIR}
)

file( GLOB APP_SOURCES *.cpp )

foreach( srcfilename ${APP_SOURCES} )
    get_filename_component(basename ${srcfilename} NAME)    
    # I used a simple string replace, to cut off .cpp.
    string( REPLACE ".cpp" "" exname ${basename} )
    add_executable( ${exname} ${srcfilename} )
    # Make sure clustering is linked to each app
    target_link_libraries( ${exname} clustering )
endforeach( srcfilename ${APP_SOURCES} )

file(COPY 
      "noisy_moons.txt"
      "comparison.py"      
    DESTINATION 
      ${CMAKE_CURRENT_BINARY_DIR})

