#targets for UT
add_executable (ann_ut_s test.c)
add_executable (ann_ut_t threaded_test.c)

#link targets to lib ann
target_link_libraries (ann_ut_s LINK_PUBLIC ann)
target_link_libraries (ann_ut_t LINK_PUBLIC ann)

