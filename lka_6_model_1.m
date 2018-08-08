function [V, phi, th] = model_1(x)

x1 = x(:,1);
x2 = x(:,2);
x3 = x(:,3);
x4 = x(:,4);
x5 = x(:,5);
x6 = x(:,6);
x7 = x(:,7);
x8 = x(:,8);
x9 = x(:,9);
x10 = x(:,10);
x11 = x(:,11);
x12 = x(:,12);
x13 = x(:,13);
x14 = x(:,14);
x15 = x(:,15);
x16 = x(:,16);
x17 = x(:,17);
x18 = x(:,18);
x19 = x(:,19);
x20 = x(:,20);
x21 = x(:,21);
x22 = x(:,22);
x23 = x(:,23);
x24 = x(:,24);

phi = [ ( ( (((x1*16.0 * 1.0 + x2*-0.31251176 * 0.63654371 + x3*1.31607401 * 0.81524363 + x4*144.0 * 1.0 + x5*-0.31251176 * 2.61145983 + x6*-155.57321356 * 1.0 + x7*56.21572441 * 0.44595561 + x8*-10.74478643 * 1.0 + x9*-63.49348587 * 1.35122157 + x10*-140.85840735 * 1.0 + x11*-155.57321356 * 1.0 + x12*-1.41421356 * 1.13722368 + x13*56.21572441 * 1.0 + x14*-10.74478643 * 1.0 + x15*48.0 * 1.49802223 + x16*12.0 * 0.80089543 + x17*115.4504355 * 1.0 + x18*12.48052971 * 1.0 + x20*43.29397388 * -0.20296562 + x21*-115.4504355 * 1.0 + x22*-10.95247279 * 1.0 + x23*1.21452931 * 1.0 + x24*-10.74478643 * 1.23388032 + 12.0 * 1.0)) - ((x1*118.764144 * 1.0 + x2*3.14159265 * 1.7186003 + x3*-63.49348587 * 1.00122334 + x4*-2.82842712 * 1.0 + x5*12.48052971 * 1.0 + x7*0.31399597 * 0.24345901 + x8*18.97579855 * 0.86034648 + x9*-0.82842712 * 1.0 + x10*75.19152296 * 0.44794263 + x11*144.0 * 1.0 + x12*56.21572441 * 1.0 + x13*-3.14159265 * 1.0 + x14*12.48052971 * 1.75641176 + x15*-10.95247279 * 1.0 + x16*-21.46584874 * 0.93268515 + x17*-21.46584874 * 0.58895026 + x18*-2.82842712 * 0.92377941 + x19*8.0 * 1.40479256 + x20*4.0 * 1.0 + x21*118.764144 * 1.0 + x22*7.49771461 * 1.0 + x23*75.19152296 * 1.0 + -13.78089991 * 0.59676976)))  -  ( power(((x1*-0.9680983 * 1.0 + x2*115.4504355 * 1.1396825 + x3*0.16495943 * 1.0 + x4*15.48224302 * 1.05597019 + x5*-3.14159265 * 1.0 + x6*-10.95247279 * 1.0 + x7*12.48052971 * 1.0 + x8*3.14159265 * 0.21106832 + x9*-10.74478643 * 1.0 + x10*13.78089991 * 1.0 + x11*144.0 * 1.0 + x12*-7.75646448 * 1.40949466 + x13*4.35612196 * 1.0 + x14*12.0 * 1.0 + x15*0.98391987 * 1.28552078 + x17*-0.40563967 * 1.0 + x18*1.11797649 * 1.58250429 + x19*3.71226345 * 1.0 + x20*-1.0 * 1.0 + x21*-7.74478643 * 1.0 + x22*0.01051992 * 1.62786803 + x23*-0.40563967 * 1.0 + x24*-2.43183169 * 0.93274589 + 0.00866173 * 0.56640403)), 2)  +  (((x1*1.0 * 1.0 + x2*9.49771461 * 1.0 + x3*346.03731054 * 1.0 + x4*-155.57321356 * 1.0 + x5*144.0 * 1.0 + x6*-0.55902751 * 1.0 + x7*-10.74478643 * 1.0 + x8*43.29397388 * 1.0 + x10*115.4504355 * 1.0 + x11*56.21572441 * 0.81370339 + x12*144.0 * 2.23538585 + x13*-7.75646448 * 1.0 + x14*12.0 * 1.0 + x15*15.48224302 * 1.19727571 + x16*-10.95247279 * 1.0 + x17*-154.74478643 * 1.83792042 + x18*-2.26794919 * 1.0 + x19*-10.74478643 * 0.99490278 + x20*-4.69978088 * 1.0 + x21*144.0 * 1.15774295 + x22*75.19152296 * 1.0 + x23*144.0 * 1.0 + x24*59.9817169 * 1.0 + -155.57321356 * 1.0)) + ((x1*3.71226345 * 0.14641119 + x2*15.48224302 * 1.52238544 + x3*75.19152296 * 1.0 + x4*-4.35612196 * 1.0 + x5*144.0 * 1.95570853 + x6*-1.41421356 * 1.0 + x7*3.14159265 * 1.05770808 + x8*56.21572441 * 1.54974187 + x9*-0.82842712 * 0.95060981 + x10*12.48052971 * 1.0 + x11*144.0 * 2.01372068 + x12*2.7381955 * 1.0 + x13*132.54504391 * 1.0 + x14*144.0 * 0.32471356 + x15*0.29289322 * 4.50078379 + x16*-110.24704508 * 1.0 + x17*-10.63930726 * 0.96940908 + x18*-144.0 * 1.0 + x20*144.0 * 1.0 + x21*14.06802945 * 1.0 + x22*132.54504391 * 1.25814276 + x23*-2.43183169 * 1.07747532 + x24*-0.16495943 * 1.03904043 + -140.85840735 * 0.71379457))) ) ) ) , ...
        ( sin( ( power(((x4*1.0 * 1.0)), 2)  - 20736.0) ) ) , ...
        ( ( (((x14*1.0 * 1.0)) +  cos(((x1*-0.82842712 * 1.0 + x2*-3.0 * 1.0 + x3*12.0 * 0.53258699 + x4*-4.69978088 * 1.26923992 + x5*0.09306839 * 1.0 + x6*0.00866173 * 1.38864186 + x8*-7.49771461 * -1.10742211 + x9*2.0 * 1.10177203 + x10*-0.82842712 * 1.0 + x11*9.49771461 * 0.63098846 + x13*144.0 * 1.97426215 + x14*-0.90563967 * 1.0 + x15*3.71226345 * 0.57613122 + x16*-7.49771461 * 1.0 + x17*2.1678978 * 2.3945451 + x18*-140.85840735 * 1.0 + x19*-63.49348587 * 2.08554244 + x20*12.0 * 1.08160806 + x21*3.23176401 * 1.0 + x22*-0.40563967 * 1.0 + x23*7.14159265 * 1.0 + x24*4.35612196 * 0.66865719 + 16.0 * 1.0))) )  .*  ( power(((x4*1.0 * 1.0)), 2)  +  cos( (((x9*1.0 * 1.0)) + ((x8*1.0 * 1.0))) ) ) ) ) , ...
        ( (((x1*0.70710678 * 1.0 + x2*-115.4504355 * 1.0 + x3*-21.46584874 * 1.09943263 + x4*0.29289322 * 1.0 + x5*1.21452931 * 1.9954692 + x6*0.31399597 * 1.0 + x7*14.06802945 * 1.0 + x8*7.49771461 * 0.92932765 + x9*7.49771461 * 1.0 + x10*-7.49771461 * 0.49367905 + x11*-12.0 * 1.48881517 + x12*-2.82842712 * 1.0 + x13*2.23595298 * 1.0 + x14*4.0 * 1.0 + x15*3.0 * 0.83073984 + x17*0.02721161 * 1.08438961 + x18*0.09306839 * 1.0 + x19*-2.26794919 * 1.0 + x20*3.0 * 1.0 + x21*3.0 * 1.0923275 + x24*-0.82842712 * 1.0 + 2.82842712 * 1.0)) .*  ( (((x3*1.0 * 1.0)) - -3.14159265)  .* -0.91973277) ) ) , ...
        (((x6*1.0 * 1.0))) , ...
        ones(length(x1), 1) ];

th = [ 4.87E-6 ; ...
       0.02993855 ; ...
       -0.00473848 ; ...
       -5.9028E-4 ; ...
       0.92870876 ; ...
       0.00178648 ];

V = phi * th;

% MSE = 0.0037812860446060142

% complexity = 51


% Configuration:
%         seed: 1
%         nbOfRuns: 5
%         dataset: Datasets_18/6_Left_Knee_Angle_1000_18.txt
%         maxGenerations: 30000
% Default nbOfThreads: 2
%         epochLength: 1000
%         maxEpochs: 30
%         populationSize: 500
%         nbOfTransformedVar: 40
%         lsIterations: 50
% Default maxNodeEvaluations: 9223372036854775807
%         depthLimit: 5
% Default probHeadNode: 0.1
% Default probTransformedVarNode: 0.2
%         useIdentityNodes: true
% Default saveModelsForPython: false
% Default optMisplacementPenalty: 0.0
% Default desiredOptimum: 
%         regressionClass: LeastSquaresFit
% Default maxWeight: 10000.0
% Default minWeight: 1.0E-20
%         populationClass: PartitionedPopulation
%         resultsDir: Results_18/6_Left_knee_Angle/5_1000_30_5_40/
%         tailFunctionSet: Multiply, Plus, Minus, Cosine, Pow2, Pow3, Sine
%         solverName: SolverMultiThreaded
%         nbRegressors: 5
%         nbPredictors: 5
% Default improvementThreshold: 0.0
% Default maxNonImprovingEpochs: 2147483647
% Default identityNodeType: identity
