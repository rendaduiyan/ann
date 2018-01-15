#targets for function test
add_executable (ann_ft_s func_test.c)
add_executable (ann_ft_t threaded_func_test.c)
add_executable (ann_ft_mnist threaded_func_test_mnist.c)

#link targets to lib ann
target_link_libraries (ann_ft_s LINK_PUBLIC ann)
target_link_libraries (ann_ft_t LINK_PUBLIC ann)
target_link_libraries (ann_ft_mnist LINK_PUBLIC ann)

