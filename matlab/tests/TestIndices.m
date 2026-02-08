classdef TestIndices < matlab.unittest.TestCase
    % Minimal sanity tests for index computations.

    methods(Test)
        function testRAIZeroForLinear(testCase)
            cfg = default_config();
            p = [0.2 0.4 0.6 0.8]; % constant rate
            r = make_ramp_event(p, 12, 'up', 0:3);
            v = calculate_rai(r);
            testCase.verifyEqual(v, 0.0, 'AbsTol', 1e-12);
        end

        function testRSCIAtLeastOne(testCase)
            cfg = default_config(); %#ok<NASGU>
            p = [0.8 0.6 0.4 0.2];
            r = make_ramp_event(p, 18, 'down', 0:3);
            v = calculate_rsci(r);
            testCase.verifyGreaterThanOrEqual(v, 1.0);
        end

        function testECSIInRange(testCase)
            cfg = default_config();
            p = [0.8 0.7 0.55 0.35 0.2];
            r = make_ramp_event(p, 18, 'down', 0:4);
            % Use batch bounds from a trivial batch
            [res, cfg2] = calculate_batch(r, cfg); %#ok<ASGLU>
            ecsi = res.ECSI;
            testCase.verifyGreaterThanOrEqual(ecsi, 0.0);
            testCase.verifyLessThanOrEqual(ecsi, 1.0);
        end
    end
end
