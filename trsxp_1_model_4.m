function [V, phi, th] = model_4(x)

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

phi = [ ( power( (((x2*1.0 * 1.0)) .* ((x1*-21.79787636 * 1.0 + x2*-327.63500988 * 1.0 + x3*311.74435454 * 1.0 + x4*-21.79787636 * 2.23776666 + x5*-58.48640807 * 1.0 + x6*27.35120622 * 1.0 + x7*77.93608863 * 0.63023092 + x9*-70.38339358 * 1.44999088 + x10*327.63500988 * 1.0 + x11*213.2542156 * 0.98798212 + x12*48.96176076 * 1.0 + x13*4.56046888 * 1.0 + x14*16.40574905 * 1.02118861 + x15*4.5 * 1.0 + x16*-17.56251348 * 1.0 + x17*-15.89065534 * 1.94188398 + x18*-0.64159265 * 1.0 + x19*-14.12502697 * 1.0 + x20*-17.56251348 * 1.0 + x21*-8.82814186 * 1.0 + x22*8.34683902 * 1.0 + x23*32.04086379 * 0.26234746 + x24*-0.5 * 1.0 + -35.31256742 * 1.43571195))) , 2) ) , ...
        ( ( ( (7.06251348 .* ((x23*1.0 * 1.0)))  +  (((x1*-21.79787636 * 1.0 + x2*-327.63500988 * 1.0 + x3*311.74435454 * 1.0 + x4*-21.79787636 * 2.23776666 + x5*-58.48640807 * 1.0 + x6*27.35120622 * 1.0 + x7*77.93608863 * 0.63023092 + x9*-70.38339358 * 1.44999088 + x10*327.63500988 * 1.0 + x11*213.2542156 * 0.98798212 + x12*48.96176076 * 1.0 + x13*4.56046888 * 1.0 + x14*16.40574905 * 1.02118861 + x15*4.5 * 1.0 + x16*-17.56251348 * 1.0 + x17*-15.89065534 * 1.94188398 + x18*-0.64159265 * 1.0 + x19*-14.12502697 * 1.0 + x20*-17.56251348 * 1.0 + x21*-8.82814186 * 1.0 + x22*8.34683902 * 1.0 + x23*32.04086379 * 0.26234746 + x24*-0.5 * 1.0 + -35.31256742 * 1.43571195)) - ((x15*1.0 * 1.0))) )  .* ((x2*1.0 * 1.0))) ) , ...
        ( ( (((x19*1.0 * 1.0)) + 213.2542156)  .*  power(((x10*1.0 * 1.0)), 2) ) ) , ...
        ( ( cos(((x2*1.0 * 1.0)))  +  sin(((x10*1.0 * 1.0))) ) ) , ...
        ( ( cos( power(((x2*1.0 * 1.0)), 2) )  -  (-52.96885114 - ((x1*1.0 * 1.0))) ) ) , ...
        ones(length(x1), 1) ];

th = [ 1.14E-6 ; ...
       1.355E-4 ; ...
       3.13E-5 ; ...
       0.02869815 ; ...
       0.99993495 ; ...
       -53.99367603 ];

V = phi * th;

% MSE = 2.0604555323863928E-6

% complexity = 47


% Configuration:
%         seed: 1
%         nbOfRuns: 5
%         dataset: Datasets_18/1_Torso_xpos_1000_18.txt
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
%         resultsDir: Results_18/1_Torso_xpos/5_1000_30_5_40/
%         tailFunctionSet: Multiply, Plus, Minus, Cosine, Pow2, Pow3, Sine
%         solverName: SolverMultiThreaded
%         nbRegressors: 5
%         nbPredictors: 5
% Default improvementThreshold: 0.0
% Default maxNonImprovingEpochs: 2147483647
% Default identityNodeType: identity
