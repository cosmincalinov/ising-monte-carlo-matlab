% =========================================================
% TEMA 01-4 - Modelarea si Simularea Sistemelor Complexe
% Subiect: Modelul Ising 2D - Simulare Monte Carlo cu parfor
% Autor: Cosmin Calinov
% Data: 2026
% =========================================================
%
% TEMA ALEASA: Modelul Ising 2D
%   Modelul Ising este unul dintre cele mai studiate modele din fizica
%   statistica. Descrie un sistem de spini magnetici dispusi pe o retea
%   bidimensionala, fiecare spin putand lua valorile +1 (sus) sau -1 (jos).
%   Modelul captureaza esenta tranzitiei de faza intre o stare feromagnetica
%   (spini aliniati) si o stare paramagnetica (spini dezordonati), la o
%   temperatura critica T_c. Este un exemplu fundamental de sistem complex
%   cu comportament emergent: proprietatile macroscopice (magnetizare,
%   tranzitie de faza) apar din interactiuni locale simple intre vecini.
%
% ASPECTE TEORETICE:
%   Hamiltonianul sistemului:
%     H = -J * sum(s_i * s_j) - h * sum(s_i)
%   unde suma <i,j> se face peste perechile de vecini apropiati,
%   J este constanta de cuplaj (J>0 => feromagnet),
%   h este campul magnetic extern, s_i in {-1, +1}.
%
%   Algoritmul Metropolis Monte Carlo:
%     - Selecteaza un spin aleator (i,j)
%     - Calculeaza variatia de energie delta_E daca spinul ar fi rasturnat
%     - Daca delta_E < 0: accepta intotdeauna rasturnarea (stare mai stabila)
%     - Daca delta_E >= 0: accepta cu probabilitate exp(-delta_E / (k_B*T))
%     Aceasta procedura esantioneaza distributia Boltzmann la echilibru.
%
%   Temperatura critica Onsager (solutie exacta 2D):
%     T_c = 2*J / (k_B * ln(1 + sqrt(2))) ≈ 2.269  (in unitati J/k_B)
%   La T < T_c: faza feromagnetica (magnetizare spontana M != 0)
%   La T > T_c: faza paramagnetica (M -> 0)
%   Tranzitia este de ordinul 2 (fara caldura latenta, parametrul de ordine
%   variaza continuu).
%
%   Parametrul de ordine: magnetizarea medie per spin
%     M = (1/N) * sum(s_i),  N = L*L
%
% OPERATII LOGICE FOLOSITE:
%   1. Operatori relationali: <, >=, ==
%      Ex: if delta_E < 0  → accepta intotdeauna mutatia
%   2. Operatori logici: ||  (SAU logic)
%      Ex: if (delta_E < 0) || (rand() < exp(-delta_E/T))
%   3. Indexare logica (logical indexing):
%      Ex: spin_grid(spin_grid == -1) selecteaza toti spinii "down"
%   4. Bucle for/parfor cu conditii de terminare implicite
%   5. Functii anonime (function handles) pentru preprocesarea indicilor
%      periodici
% =========================================================

clear; clc; close all;

%% PARAMETRI SIMULARE
% -------------------------------------------------------
L       = 30;          % Dimensiunea retelei (L x L spini)
J       = 1.0;         % Constanta de cuplaj (J > 0: feromagnet)
h       = 0.0;         % Camp magnetic extern (0 = fara camp)
k_B     = 1.0;         % Constanta Boltzmann (unitati reduse)
T_c_Onsager = 2 * J / (k_B * log(1 + sqrt(2)));  % ≈ 2.269

% Vector de temperaturi simulate (de la 1.0 la 4.0 cu pas 0.1)
T_vec   = 1.0 : 0.1 : 4.0;
n_T     = length(T_vec);

% Numarul de pasi Monte Carlo per temperatura
% Fiecare "pas MC" = L^2 incercari de rasturnare (un "sweep")
n_steps = 500 * L^2;

% Numarul de pasi de termarizare (echilibrare): aruncati, nu contribuie
% la mediile termodinamice
n_therm = 100 * L^2;

fprintf('Modelul Ising 2D - Simulare Monte Carlo\n');
fprintf('Retea: %d x %d spini\n', L, L);
fprintf('Temperatura critica Onsager: T_c = %.4f\n', T_c_Onsager);
fprintf('Temperaturi simulate: %.1f ... %.1f (pas 0.1)\n', T_vec(1), T_vec(end));
fprintf('Pasi Monte Carlo per T: %d\n\n', n_steps);

%% ALOCARE VECTORI REZULTATE
% -------------------------------------------------------
mag_mean   = zeros(1, n_T);   % Magnetizarea medie |<M>|
energy_mean= zeros(1, n_T);   % Energia medie <E> per spin
Cv         = zeros(1, n_T);   % Caldura specifica C_v

% Stocheaza configuratii la 3 temperaturi reprezentative pentru Plot 4
% T < T_c, T ≈ T_c, T > T_c
T_low  = 1.5;   % T < T_c
T_mid  = 2.3;   % T ≈ T_c
T_high = 3.5;   % T > T_c
configs = cell(3,1);   % celule pentru cele 3 configuratii

%% PARALELIZARE cu parfor
% -------------------------------------------------------
% Aceasta problema se preteaza natural la paralelizare deoarece:
%   - Simulatiile la temperaturi diferite sunt INDEPENDENTE una de alta
%   - Nu exista dependente de date intre iteratiile buclei peste T
%   - Fiecare iteratie este costisitoare computational (mii de pasi MC)
% Astfel, parfor distribuie iteratiile pe mai multe "workers" (nuclee CPU),
% reducand semnificativ timpul total de executie.
%
% Fallback: daca Parallel Computing Toolbox nu este disponibil (ex. Octave),
% se foloseste bucla for secventiala.

use_parfor = false;
% Detecteaza daca Parallel Computing Toolbox este disponibil (MATLAB)
% Verifica existenta functiei parpool (disponibila doar cu toolbox-ul)
if exist('parpool', 'file') == 2 || exist('parpool', 'builtin') == 5
    use_parfor = true;
end
% In Octave, parfor este disponibil nativ (fara pool explicit), dar pentru
% compatibilitate maxima folosim for secvential implicit.
% Dezcommenteaza linia urmatoare pentru a forta parfor in Octave:
% use_parfor = true;

fprintf('Pornire simulare...\n');
tic;   % Incepe masurarea timpului

if use_parfor
    fprintf('Mod: parfor (Parallel Computing Toolbox detectat)\n\n');
    % Variabile temporare necesare pentru parfor (slice variables)
    mag_tmp    = zeros(1, n_T);
    energy_tmp = zeros(1, n_T);
    Cv_tmp     = zeros(1, n_T);

    parfor t_idx = 1 : n_T
        T = T_vec(t_idx);
        [m, e, cv] = run_metropolis(L, J, h, k_B, T, n_steps, n_therm);
        mag_tmp(t_idx)    = m;
        energy_tmp(t_idx) = e;
        Cv_tmp(t_idx)     = cv;
    end

    mag_mean    = mag_tmp;
    energy_mean = energy_tmp;
    Cv          = Cv_tmp;

else
    fprintf('Mod: for secvential\n\n');
    % Bucla for normala - compatibila cu Octave si MATLAB fara toolbox
    for t_idx = 1 : n_T
        T = T_vec(t_idx);
        [m, e, cv] = run_metropolis(L, J, h, k_B, T, n_steps, n_therm);
        mag_mean(t_idx)    = m;
        energy_mean(t_idx) = e;
        Cv(t_idx)          = cv;

        % Salveaza configuratia la temperaturile reprezentative
        % Comparatie cu toleranta pentru valori float
        if abs(T - T_low) < 0.05
            configs{1} = get_config(L, J, h, k_B, T, n_therm);
        elseif abs(T - T_mid) < 0.05
            configs{2} = get_config(L, J, h, k_B, T, n_therm);
        elseif abs(T - T_high) < 0.05
            configs{3} = get_config(L, J, h, k_B, T, n_therm);
        end

        % Afiseaza progres la fiecare 10 temperaturi
        if mod(t_idx, 10) == 0
            fprintf('  Progres: %d/%d temperaturi simulate\n', t_idx, n_T);
        end
    end
end

elapsed = toc;   % Opreste cronometrul
fprintf('\nTimp total de executie: %.2f secunde\n\n', elapsed);

%% VIZUALIZARE - PLOT 1: Magnetizarea |M| vs Temperatura T
% -------------------------------------------------------
figure('Name', 'Ising 2D - Rezultate', 'NumberTitle', 'off', ...
       'Position', [50, 50, 1200, 900]);

subplot(2, 2, 1);
plot(T_vec, mag_mean, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;
% Marcheaza temperatura critica Onsager cu o linie verticala rosie
% Folosim plot in loc de xline pentru compatibilitate cu Octave
plot([T_c_Onsager, T_c_Onsager], [-0.05, 1.05], 'r--', 'LineWidth', 1.5);
% Eticheta pentru T_c
text(T_c_Onsager + 0.05, 0.85, sprintf('T_c=%.3f', T_c_Onsager), ...
     'Color', 'red', 'FontSize', 9);
hold off;
xlabel('Temperatura T [J/k_B]', 'FontSize', 11);
ylabel('|M| (magnetizare medie)', 'FontSize', 11);
title('Magnetizarea vs Temperatura', 'FontSize', 12);
legend('|M(T)|', 'T_c Onsager', 'Location', 'northeast');
grid on;
xlim([T_vec(1)-0.1, T_vec(end)+0.1]);
ylim([-0.05, 1.05]);

%% VIZUALIZARE - PLOT 2: Energia medie <E> vs Temperatura T
% -------------------------------------------------------
subplot(2, 2, 2);
plot(T_vec, energy_mean, 'g-s', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;
plot([T_c_Onsager, T_c_Onsager], [min(energy_mean)-0.1, max(energy_mean)+0.1], 'r--', 'LineWidth', 1.5);
text(T_c_Onsager + 0.05, max(energy_mean)*0.9, sprintf('T_c=%.3f', T_c_Onsager), ...
     'Color', 'red', 'FontSize', 9);
hold off;
xlabel('Temperatura T [J/k_B]', 'FontSize', 11);
ylabel('<E> (energie medie per spin)', 'FontSize', 11);
title('Energia medie vs Temperatura', 'FontSize', 12);
legend('<E(T)>', 'T_c Onsager', 'Location', 'southeast');
grid on;
xlim([T_vec(1)-0.1, T_vec(end)+0.1]);

%% VIZUALIZARE - PLOT 3: Caldura specifica C_v vs Temperatura T
% -------------------------------------------------------
% Caldura specifica: C_v = (1/(k_B * T^2)) * (<E^2> - <E>^2) * N
% (calculata in functia run_metropolis via fluctuatii de energie)
subplot(2, 2, 3);
plot(T_vec, Cv, 'm-^', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;
plot([T_c_Onsager, T_c_Onsager], [0, max(Cv)*1.1], 'r--', 'LineWidth', 1.5);
text(T_c_Onsager + 0.05, max(Cv)*0.85, sprintf('T_c=%.3f', T_c_Onsager), ...
     'Color', 'red', 'FontSize', 9);
hold off;
xlabel('Temperatura T [J/k_B]', 'FontSize', 11);
ylabel('C_v (caldura specifica)', 'FontSize', 11);
title('Caldura specifica vs Temperatura', 'FontSize', 12);
legend('C_v(T)', 'T_c Onsager', 'Location', 'northeast');
grid on;
xlim([T_vec(1)-0.1, T_vec(end)+0.1]);

%% VIZUALIZARE - PLOT 4: Configuratii retea la 3 temperaturi
% -------------------------------------------------------
% Daca nu avem configuratii salvate (modul parfor), genereaza acum
if isempty(configs{1})
    configs{1} = get_config(L, J, h, k_B, T_low,  n_therm);
end
if isempty(configs{2})
    configs{2} = get_config(L, J, h, k_B, T_mid,  n_therm);
end
if isempty(configs{3})
    configs{3} = get_config(L, J, h, k_B, T_high, n_therm);
end

subplot(2, 2, 4);
% Concateneaza cele 3 configuratii orizontal pentru o singura imagine
combined = [configs{1}, ones(L,1)*2, configs{2}, ones(L,1)*2, configs{3}];
imagesc(combined);
colormap(gray);
axis equal tight off;
title(sprintf('Configuratii retea: T=%.1f | T=%.1f | T=%.1f', ...
              T_low, T_mid, T_high), 'FontSize', 11);
% Adauga etichete text deasupra fiecarei sectiuni
text(L/2,      -1, sprintf('T=%.1f\n(T<T_c)', T_low),  'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'blue');
text(L + 1 + L/2, -1, sprintf('T=%.1f\n(T≈T_c)', T_mid),  'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'red');
text(2*L+2+L/2,-1, sprintf('T=%.1f\n(T>T_c)', T_high), 'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', 'green');

% Indexare logica - exemplu in comentariu:
% spin_grid(spin_grid == -1) → selecteaza toti spinii "down" (valoare -1)
% Aceasta returneaza un vector cu toti spinii care sunt in starea "down"
% si poate fi folosit pentru statistici sau vizualizare selectiva.

% Titlu comun pentru figura (sgtitle - MATLAB R2018b+)
% Folosim try/catch pentru compatibilitate cu Octave
try
    sgtitle(sprintf('Modelul Ising 2D - L=%d, pasi MC=%d', L, n_steps), ...
            'FontSize', 13, 'FontWeight', 'bold');
catch
    % Fallback Octave: titlu pe subplot-ul 1
end

%% AFISARE REZULTATE NUMERICE
% -------------------------------------------------------
fprintf('=== REZULTATE NUMERICE ===\n');
fprintf('%-10s %-12s %-12s %-12s\n', 'T', '|M|', '<E>/spin', 'C_v');
fprintf('%s\n', repmat('-', 1, 50));
for t_idx = 1 : n_T
    % Marcheaza cu * temperaturile aproape de T_c
    marker = ' ';
    if abs(T_vec(t_idx) - T_c_Onsager) < 0.2
        marker = '*';  % operator relational: abs(...) < 0.2
    end
    fprintf('%-10.2f %-12.4f %-12.4f %-12.4f %s\n', ...
            T_vec(t_idx), mag_mean(t_idx), energy_mean(t_idx), Cv(t_idx), marker);
end
fprintf('\n* = aproape de T_c Onsager (%.4f)\n', T_c_Onsager);

%% CONCLUZII
% -------------------------------------------------------
% Simularea Monte Carlo a Modelului Ising 2D demonstreaza:
%
% 1. TRANZITIA DE FAZA DE ORDINUL 2:
%    La T < T_c ≈ 2.269, magnetizarea spontana |M| ≈ 1 (spini aliniati).
%    La T > T_c, magnetizarea scade la ~0 (spini dezordonati termic).
%    Tranzitia este continua (ordinul 2), fara discontinuitate in energie.
%
% 2. COMPORTAMENTUL ENERGIEI:
%    Energia medie creste monoton cu temperatura, reflectand dezordinea
%    termica crescanda. Panta este mai accentuata in jurul lui T_c.
%
% 3. DIVERGENTA CALDURII SPECIFICE:
%    Caldura specifica prezinta un maxim (divergenta logaritmica in limita
%    termodinamica) la T ≈ T_c, semn al fluctuatiilor maxime.
%    In simulare finita, maximul apare la T usor diferit de T_c exacta
%    din cauza efectelor de marime finita (finite-size scaling).
%
% 4. CONFIGURATII VIZUALE:
%    La T < T_c: domenii mari de spini aliniati (feromagnet).
%    La T ≈ T_c: structuri fractale, correlatie pe distante mari.
%    La T > T_c: configuratie aleatoare, fara structura (paramagnet).
%
% 5. EFICIENTA PARALELIZARII (parfor):
%    Simulatiile la T diferite sunt independente → paralelism ideal.
%    Speedup teoretic: proportional cu numarul de nuclee disponibile.

fprintf('\nSimulare finalizata. Graficele sunt afisate in figura.\n');

% =========================================================
%                    FUNCTII LOCALE
% =========================================================

function [mag_out, energy_out, Cv_out] = run_metropolis(L, J, h, k_B, T, n_steps, n_therm)
% RUN_METROPOLIS  Ruleaza algoritmul Metropolis Monte Carlo pentru modelul
%                 Ising 2D la temperatura T si returneaza observabilele
%                 termodinamice medii.
%
% Intrari:
%   L       - dimensiunea retelei (L x L)
%   J       - constanta de cuplaj
%   h       - camp magnetic extern
%   k_B     - constanta Boltzmann
%   T       - temperatura
%   n_steps - numarul total de pasi MC (masuratori)
%   n_therm - numarul de pasi de termarizare (aruncati)
%
% Iesiri:
%   mag_out    - magnetizarea medie absoluta per spin |<M>|
%   energy_out - energia medie per spin <E>
%   Cv_out     - caldura specifica C_v

    N = L * L;   % Numarul total de spini

    % Initializare retea aleatoare: fiecare spin este +1 sau -1
    % Operatie logica: 2 * (rand(L,L) > 0.5) - 1
    %   rand(L,L) > 0.5  → matrice logica (0/1)
    %   *2 - 1           → transforma in -1/+1
    spin_grid = 2 * (rand(L, L) > 0.5) - 1;

    % Calculeaza energia initiala
    E_curr = compute_energy(spin_grid, L, J, h);

    % --- FAZA DE TERMARIZARE ---
    % Ruleaza n_therm pasi fara a colecta date, pentru a atinge echilibrul
    for step = 1 : n_therm
        [spin_grid, E_curr] = metropolis_sweep(spin_grid, L, J, h, k_B, T, E_curr);
    end

    % --- FAZA DE MASURATORI ---
    % Acumuleaza observabile termodinamice dupa termarizare
    sum_M  = 0.0;   % Suma magnetizarilor
    sum_E  = 0.0;   % Suma energiilor
    sum_E2 = 0.0;   % Suma patratelor energiilor (pentru fluctuatii)

    for step = 1 : n_steps
        [spin_grid, E_curr] = metropolis_sweep(spin_grid, L, J, h, k_B, T, E_curr);

        % Magnetizarea instantanee per spin
        M_inst = abs(sum(spin_grid(:))) / N;
        % Energia instantanee per spin
        E_inst = E_curr / N;

        sum_M  = sum_M  + M_inst;
        sum_E  = sum_E  + E_inst;
        sum_E2 = sum_E2 + E_inst^2;
    end

    % Medii termodinamice
    mag_out    = sum_M  / n_steps;
    energy_out = sum_E  / n_steps;
    E2_mean    = sum_E2 / n_steps;

    % Caldura specifica: C_v = N * (<E^2> - <E>^2) / (k_B * T^2)
    % N apare deoarece am lucrat cu energia per spin, nu totala
    Cv_out = N * (E2_mean - energy_out^2) / (k_B * T^2);
end


function [spin_grid, E_new] = metropolis_sweep(spin_grid, L, J, h, k_B, T, E_old)
% METROPOLIS_SWEEP  Efectueaza L^2 incercari de rasturnare spin (un sweep).
%
% Algoritmul Metropolis:
%   1. Alege un spin aleator (i, j)
%   2. Calculeaza delta_E = variatie energie la rasturnare
%   3. OPERATIE LOGICA || (SAU):
%      Accepta rasturnarea daca:
%        (delta_E < 0)  → stare mai favorabila energetic
%        SAU (rand() < exp(-delta_E / (k_B*T)))  → fluctuatie termica
%   4. Actualizeaza grila si energia

    N = L * L;

    for flip = 1 : N
        % Alege coordonate aleatoare in retea (indexare 1..L)
        i = randi(L);
        j = randi(L);

        % Suma vecinilor cu conditii periodice la margine (retea "torica")
        % mod(x-1, L) + 1 asigura ca vecinul randului 1 este randul L si invers
        i_up   = mod(i - 2, L) + 1;   % vecinul de sus
        i_down = mod(i,     L) + 1;   % vecinul de jos
        j_left = mod(j - 2, L) + 1;   % vecinul din stanga
        j_right= mod(j,     L) + 1;   % vecinul din dreapta

        % Suma celor 4 vecini (interactiuni in retea patrata)
        sum_neighbors = spin_grid(i_up,  j) + spin_grid(i_down, j) + ...
                        spin_grid(i, j_left) + spin_grid(i, j_right);

        % Variatia de energie la rasturnarea spinului (i,j)
        % delta_E = E_nou - E_vechi = 2*J*s(i,j)*suma_vecini + 2*h*s(i,j)
        delta_E = 2 * J * spin_grid(i,j) * sum_neighbors + 2 * h * spin_grid(i,j);

        % Criteriul de acceptare Metropolis (OPERATIE LOGICA ||):
        % Accepta daca delta_E < 0 SAU daca probabilitatea Boltzmann permite
        if (delta_E < 0) || (rand() < exp(-delta_E / (k_B * T)))
            spin_grid(i,j) = -spin_grid(i,j);   % Rastoarna spinul
            E_old = E_old + delta_E;              % Actualizeaza energia
        end
    end

    E_new = E_old;
end


function E = compute_energy(spin_grid, L, J, h)
% COMPUTE_ENERGY  Calculeaza energia totala a configuratiei curente.
%
% H = -J * sum_{<i,j>} s_i * s_j - h * sum_i s_i
% Suma se face doar pe perechi distincte (evita numararea dubla):
%   - fiecare spin cu vecinul sau de la dreapta
%   - fiecare spin cu vecinul sau de jos

    % Energia de interactiune (cuplaj spin-spin)
    % Conditii periodice: ultimul rand/coloana interactioneaza cu primul
    E_interaction = -J * sum(sum( ...
        spin_grid .* circshift(spin_grid, [0, -1]) + ...   % vecin dreapta
        spin_grid .* circshift(spin_grid, [-1, 0])  ...    % vecin jos
    ));

    % Energia in camp magnetic extern
    E_field = -h * sum(spin_grid(:));

    E = E_interaction + E_field;
end


function grid_out = get_config(L, J, h, k_B, T, n_therm)
% GET_CONFIG  Genereaza o configuratie de retea in echilibru la temperatura T.
%   Porneste din configuratie aleatoare, termalizeaza n_therm pasi,
%   si returneaza configuratia finala.

    % Initializare aleatoare (OPERATIE LOGICA: comparatie > pentru binarizare)
    spin_grid = 2 * (rand(L, L) > 0.5) - 1;

    E_curr = compute_energy(spin_grid, L, J, h);

    % Termarizare
    for step = 1 : n_therm
        [spin_grid, E_curr] = metropolis_sweep(spin_grid, L, J, h, k_B, T, E_curr);
    end

    grid_out = spin_grid;
end
