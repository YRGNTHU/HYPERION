%clear all; close all; clc
%% setting
N=5;
lambda= 1;
mixture_index = [2 3 5 7 9 10 11 13 14 15];

%load data
load("./data/N5.mat");
X = X.';

%Generate the dataset without pure spectra
X = X(:, mixture_index);
%% agorithm
tic
[Signatures,T,time] = HYPERION(X,N,lambda);
toc

%plot the unmixed spectra
plot(Signatures);
