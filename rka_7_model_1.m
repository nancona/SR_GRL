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

phi = [ (((x1*-0.01023026 * 1.0 + x2*10.62677172 * 1.0 + x3*-58.50941486 * 1.0 + x4*-0.01023026 * 1.0 + x5*-6.14159265 * 1.28817822 + x6*4.43480219 * 1.0 + x7*2.22144147 * 1.0 + x8*19.50313829 * 1.19273063 + x9*-13.27795693 * 1.48094413 + x10*0.5 * 0.53667126 + x12*8.72606823 * 1.0 + x15*-0.5 * 1.0 + x16*-2.47766021 * 1.39042318 + x17*-0.1767767 * 1.0 + x18*-2.0 * 1.0 + x19*-1.41421356 * 1.0 + x21*0.5 * 1.31043808 + x22*-3.0 * 0.90133911 + x23*0.45015816 * 1.21664784 + x24*0.70710678 * 1.13482849 + 6.0 * 1.0))) , ...
        ( ( (((x15*1.0 * 1.0)) - 15.50313829)  .*  cos(((x5*1.0 * 1.0))) ) ) , ...
        ( ( ( ( (((x1*4.0 * 1.0 + x2*9.86960438 * 0.78070472 + x3*-2.0 * 1.09816637 + x4*3.0 * 1.57183849 + x5*-4.0 * 0.75995945 + x6*-13.50525941 * 1.11676954 + x7*12.09104585 * 1.0 + x8*-17.45213647 * -0.70706204 + x9*-58.50941486 * 1.0 + x10*-74.01255315 * 0.75502795 + x11*9.86960438 * 1.0 + x12*-17.45213647 * 1.0 + x13*-0.24763524 * 1.0 + x14*1.41421356 * 1.0 + x15*9.86960438 * 1.0 + x16*-1.41421356 * 0.95837204 + x17*-3.0 * -0.16460564 + x18*-7.41981208 * 1.0 + x19*4.0 * 1.0 + x20*4.0 * 1.37561148 + x21*-2.22144147 * 1.0 + x22*1.41421356 * 1.0 + x23*3.0 * 1.0 + x24*6.0 * 1.0 + 2.0 * 1.0)) - -2.0)  .* 1.22778017)  .*  power(((x2*1.0 * 1.0)), 3) )  +  ( ( power(((x3*1.0 * 1.0)), 2)  +  sin(((x4*1.0 * 1.0))) )  .* 0.20698401) ) ) , ...
        ( cos( cos( (2.22144147 + ((x7*1.0 * 1.0))) ) ) ) , ...
        ( (11.42000145 .*  (2.22144147 + ((x7*1.0 * 1.0))) ) ) , ...
        ones(length(x1), 1) ];

th = [ -0.00191899 ; ...
       0.0021481 ; ...
       -0.55871159 ; ...
       0.39132314 ; ...
       0.07700929 ; ...
       -2.34700896 ];

V = phi * th;

% MSE = 0.004851020529096153

% complexity = 49


% Configuration:
%         seed: 1
%         nbOfRuns: 5
%         dataset: Datasets_18/7_Right_Knee_Angle_1000_18.txt
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
%         resultsDir: Results_18/7_Right_Knee_Angle/5_1000_30_5_40/
%         tailFunctionSet: Multiply, Plus, Minus, Cosine, Pow2, Pow3, Sine
%         solverName: SolverMultiThreaded
%         nbRegressors: 5
%         nbPredictors: 5
% Default improvementThreshold: 0.0
% Default maxNonImprovingEpochs: 2147483647
% Default identityNodeType: identity
