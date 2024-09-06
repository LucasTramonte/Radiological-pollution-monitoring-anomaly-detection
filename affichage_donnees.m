clear; close all
% Donnees
% Choix du mois (n° de colonne du fichier)
mois=3; %1: fevrier,2 :avril, 3: juin, 4:octobre

% données gamma
filename = '2015_months_DebitDoseA.txt'; 
delimiterIn = ',';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);
sigg=A.data(:, mois);  

% données temperature
filename = '2015_months_TEMP.txt'; 
delimiterIn = ',';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);
sigt=A.data(:, mois); 

% données hygrometrie
filename = '2015_months_HYGR.txt'; 
delimiterIn = ',';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);
sigh=A.data(:, mois); 

% données pression atmospherique
filename = '2015_months_PATM.txt'; 
delimiterIn = ',';
headerlinesIn = 1;
A = importdata(filename,delimiterIn,headerlinesIn);
sigp=A.data(:, mois); 


%affichage des donnees : on elimine la premiere et la dernière heure
N=length(sigg); % nombre de points des signaux
fe=1/60;
t=[0:N-1];
Nh=60;  % 1h
figure
plot(t(Nh:N-Nh),sigg(Nh:N-Nh))
hold on
plot(t(Nh:N-Nh),sigt(Nh:N-Nh))
plot(t(Nh:N-Nh),sigh(Nh:N-Nh))
plot(t(Nh:N-Nh),sigp(Nh:N-Nh)-800)
xlabel('temps en min')
title('donnees')
legend('gamma','temperature','hygrometrie','pression -800')



