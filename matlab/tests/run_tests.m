% RUN_TESTS Execute MATLAB unit tests.
% From matlab/ folder:
%   run('tests/run_tests.m')
import matlab.unittest.TestSuite
suite = TestSuite.fromFolder(pwd);
results = run(suite);
disp(results);
