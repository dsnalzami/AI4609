function [Patterns,Targets,V_Targets]=LoadData(dataset,scale)
dataset=[char(dataset),'.txt'];
RawData=load(strcat('datasets/',dataset));
Patterns=RawData(:,1:end-1);
Targets=RawData(:,end);

if min(Targets)==0, Targets = Targets+1; end

N_Samples=size(Patterns,1);
N_class=max(Targets);

%% create vector target (V_Target)
% V_target are used in classifier: ML
% description: if the target = 2, and we have 3 classes
%              the data become V_target = [0 1 0]
V_Targets=zeros(N_class,N_Samples);
temp=0:N_class:(N_Samples-1)*N_class;
V_Targets(Targets'+temp)=1;

if scale==1,
    Patterns=mapminmax(Patterns'); Patterns=Patterns';
end;
Patterns=removeconstantrows(Patterns'); Patterns=Patterns';
end