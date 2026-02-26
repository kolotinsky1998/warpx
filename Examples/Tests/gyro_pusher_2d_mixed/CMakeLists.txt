# Add tests ##############################################################
#

add_warpx_test(
    test_2d_gyro_pusher_mixed  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_gyro_pusher_mixed  # inputs
    "analysis.py diags/diag1000020"  # analysis
    "analysis.py diags/diag1000020"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_gyro_pusher_invalid_b  # name
    2  # dims
    1  # nprocs
    inputs_test_2d_gyro_pusher_invalid_b  # inputs
    "analysis_noop.py"  # analysis
    "analysis_noop.py"  # checksum
    OFF  # dependency
)

if (TEST test_2d_gyro_pusher_invalid_b.run)
    set_tests_properties(test_2d_gyro_pusher_invalid_b.run PROPERTIES WILL_FAIL TRUE)
endif()
