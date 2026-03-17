% =========================================================
% TEMA 01-4 - Modelarea si Simularea Sistemelor Complexe
% Subiect: Estimarea lui π prin metoda Monte Carlo cu parfor
% Autor: Cosmin Calinov
% Data: 2026
% =========================================================
%
% TEMA ALEASA: Estimarea lui π prin metoda Monte Carlo
%
%   Metoda Monte Carlo foloseste esantionare aleatoare pentru a estima
%   valori matematice. In acest script, estimam constanta π exploatand
%   o proprietate geometrica simpla:
%
%   PRINCIPIUL GEOMETRIC:
%     - Consideram un cerc de raza r=1 inscris intr-un patrat 2x2.
%     - Aria cercului: A_cerc = π * r^2 = π
%     - Aria patratului: A_patrat = (2r)^2 = 4
%     - Raportul ariilor: A_cerc / A_patrat = π / 4
%
%   ESTIMATORUL MONTE CARLO:
%     - Generam N puncte (x, y) uniform distribuite in [-1,1] x [-1,1]
%     - Un punct este "in cerc" daca x^2 + y^2 <= 1
%     - Prin Legea Numerelor Mari: Nr_in_cerc / N → π / 4 (cand N → ∞)
%     - Deci: π ≈ 4 * Nr_in_cerc / N
%
%   CONVERGENTA (Legea Numerelor Mari):
%     - Eroarea standard scade proportional cu 1/sqrt(N)
%     - Dublarea preciziei necesita de 4 ori mai multe puncte
%     - Este o metoda generala: functioneaza si in dimensiuni inalte (C4)
%
%   LEGATURA CU INTEGRAREA NUMERICA MONTE CARLO (C4 din curs):
%     - π/4 = Integral pe [-1,1]x[-1,1] al functiei indicator {x^2+y^2<=1}
%     - Monte Carlo evalueaza aceasta integrala prin medie empirica
%     - Avantajul fata de metode clasice: complexitatea nu creste exponential
%       cu dimensiunea spatiului (nu sufera de "blestemul dimensionalitatii")
%     - Aplicatii practice: integrare in fizica, finante, machine learning
%
% OPERATII LOGICE FOLOSITE:
%   1. Operatori relationali: <=
%      Ex: in_circle = (x.^2 + y.^2) <= 1  → punct in interiorul cercului
%   2. Indexare logica (logical indexing):
%      Ex: x(in_circle)  → coordonatele x ale punctelor DIN cerc
%      Ex: x(~in_circle) → coordonatele x ale punctelor IN AFARA cercului
%   3. Operatori logici: ~ (negatie logica)
%   4. sum() pe matrice logica: sum(in_circle) = nr. puncte in cerc
%   5. parfor cu reducere (acumulare rezultate independente)
%
% =========================================================

clear; clc; close all;

%% PARAMETRI
% -------------------------------------------------------
N_vec = [100, 1000, 10000, 100000, 1000000, 10000000];  % dimensiuni esantion
n_rep = 20;   % numar de repetari per dimensiune (pentru statistici)

% Detecteaza daca Parallel Computing Toolbox este disponibil
use_parfor = false;
if exist('parpool', 'file') == 2 || exist('parpool', 'builtin') == 5
    use_parfor = true;
end

%% SECTIUNEA 1: DEMONSTRATIE VIZUALA (N = 10000 puncte)
% -------------------------------------------------------
% Scopul: ilustrarea intuitiva a metodei Monte Carlo pentru estimarea lui π.
% Generam puncte aleatoare in patratul [-1,1]x[-1,1] si coloram diferit
% cele care cad in interiorul cercului de raza 1.

N_demo = 10000;   % numar de puncte pentru demonstratia vizuala

% Genereaza puncte uniforme in [-1, 1] x [-1, 1]
x = 2 * rand(N_demo, 1) - 1;   % coordonate x uniforme in [-1, 1]
y = 2 * rand(N_demo, 1) - 1;   % coordonate y uniforme in [-1, 1]

% OPERATIE LOGICA: operator relational <= produce un vector logic
% in_circle(i) = true daca punctul i se afla in sau pe cercul unitate
in_circle = (x.^2 + y.^2) <= 1;

% Calculeaza estimarea lui π pentru aceasta demonstratie
% sum() aplicat unui vector logic numara elementele true
pi_demo = 4 * sum(in_circle) / N_demo;

% Creaza figura pentru demonstratia vizuala
figure('Name', 'Demonstratie Monte Carlo π', 'NumberTitle', 'off', ...
       'Position', [100 100 600 600]);

hold on;

% INDEXARE LOGICA: x(in_circle) selecteaza doar punctele DIN cerc
% Punctele din interiorul cercului - afisate in albastru
scatter(x(in_circle),  y(in_circle),  1, [0.2 0.4 0.8], 'filled');

% INDEXARE LOGICA CU NEGATIE: x(~in_circle) selecteaza punctele IN AFARA
% Punctele din exteriorul cercului - afisate in rosu
scatter(x(~in_circle), y(~in_circle), 1, [0.8 0.2 0.2], 'filled');

% Deseneaza cercul de raza 1 (conturul geometric de referinta)
theta = linspace(0, 2*pi, 500);    % 500 de unghiuri pentru un cerc neted
plot(cos(theta), sin(theta), 'k-', 'LineWidth', 2);

% Deseneaza patratul [-1,1] x [-1,1]
rectangle('Position', [-1 -1 2 2], 'EdgeColor', 'k', 'LineWidth', 2);

axis equal;   % aspect ratio egal pentru a vedea cercul corect
axis([-1.05 1.05 -1.05 1.05]);

title(sprintf('Monte Carlo: \\pi \\approx %.4f  (N = %d)', pi_demo, N_demo), ...
      'FontSize', 14);
xlabel('x');
ylabel('y');
legend({'In cerc (albastru)', 'In afara (rosu)', 'Cercul r=1'}, ...
       'Location', 'SouthEast');
grid on;
hold off;

fprintf('Demonstratie vizuala generata (N = %d puncte).\n', N_demo);

%% SECTIUNEA 2: CONVERGENTA ESTIMARII cu parfor
% -------------------------------------------------------
% Pentru fiecare valoare N din N_vec, rulam n_rep estimari independente
% (parfor distribuie aceste repetari pe mai multe nuclee CPU).
% Calculam media si deviatia standard a estimarilor pentru a ilustra
% convergenta: eroarea scade proportional cu 1/sqrt(N).

fprintf('\n=== ESTIMAREA LUI π prin Monte Carlo ===\n');

n_N    = length(N_vec);           % numar de dimensiuni testate
pi_mean = zeros(1, n_N);          % media estimarilor pentru fiecare N
pi_std  = zeros(1, n_N);          % deviatia standard
pi_err  = zeros(1, n_N);          % eroarea absoluta medie

for k = 1 : n_N
    N     = N_vec(k);             % dimensiunea curenta a esantionului
    pi_est = zeros(1, n_rep);     % vector pentru cele n_rep estimari

    if use_parfor
        % parfor: repetarile sunt INDEPENDENTE intre ele (nu exista
        % dependente de date), deci se preteaza perfect la paralelizare.
        % Fiecare worker calculeaza estimarea pentru o repetare diferita.
        parfor r = 1 : n_rep
            xr = 2 * rand(N, 1) - 1;   % coordonate x aleatoare
            yr = 2 * rand(N, 1) - 1;   % coordonate y aleatoare
            % OPERATIE LOGICA: operator <= produce vector logic
            % sum() pe vectorul logic numara punctele din cerc
            pi_est(r) = 4 * sum(xr.^2 + yr.^2 <= 1) / N;
        end
    else
        % for secvential - fallback cand toolbox-ul nu e disponibil
        for r = 1 : n_rep
            xr = 2 * rand(N, 1) - 1;   % coordonate x aleatoare
            yr = 2 * rand(N, 1) - 1;   % coordonate y aleatoare
            pi_est(r) = 4 * sum(xr.^2 + yr.^2 <= 1) / N;
        end
    end

    % Calculeaza statisticile pentru aceasta valoare a lui N
    pi_mean(k) = mean(pi_est);           % media estimarilor
    pi_std(k)  = std(pi_est);            % deviatia standard
    pi_err(k)  = abs(pi_mean(k) - pi); % eroarea absoluta fata de π exact

    % Afiseaza rezultatele in Command Window
    fprintf('N = %10d:  π ≈ %.5f  (eroare: %.5f, std: %.4f)\n', ...
            N, pi_mean(k), pi_err(k), pi_std(k));
end

fprintf('\nValoarea exacta: π = %.14f...\n', pi);

% Plot convergenta: π estimat cu bare de eroare vs log10(N)
figure('Name', 'Convergenta estimarii lui π', 'NumberTitle', 'off', ...
       'Position', [720 100 700 500]);

errorbar(log10(N_vec), pi_mean, pi_std, 'bo-', ...
         'LineWidth', 1.5, 'MarkerSize', 6, 'CapSize', 8);
hold on;

% Linie orizontala la valoarea exacta a lui π pentru referinta
yline(pi, 'r--', 'LineWidth', 2, 'Label', '\pi exact');

xlabel('log_{10}(N)  [numarul de puncte]');
ylabel('π estimat');
title('Convergenta estimarii Monte Carlo a lui π (cu bare de eroare ± σ)');
legend({'π estimat ± σ', 'π exact'}, 'Location', 'East');
grid on;
xlim([log10(N_vec(1))-0.5, log10(N_vec(end))+0.5]);
hold off;

%% SECTIUNEA 3: COMPARATIE SERIAL vs PARALEL
% -------------------------------------------------------
% Masuram si comparam timpul de executie al buclei for (serial) vs
% parfor (paralel) pentru un N mare si n_rep repetari.
% Speedup-ul depinde de numarul de nuclee disponibile.

N_timing = 1000000;    % N mare pentru a face diferenta de timp vizibila
n_rep_timing = 20;     % numar de repetari pentru timing

fprintf('\n=== COMPARATIE SERIAL vs PARALEL ===\n');

% --- Masoara timpul serial (bucla for) ---
pi_serial = zeros(1, n_rep_timing);
tic;   % porneste cronometrul
for r = 1 : n_rep_timing
    xs = 2 * rand(N_timing, 1) - 1;
    ys = 2 * rand(N_timing, 1) - 1;
    pi_serial(r) = 4 * sum(xs.^2 + ys.^2 <= 1) / N_timing;
end
t_serial = toc;   % opreste cronometrul, salveaza timpul

% --- Masoara timpul paralel (bucla parfor) ---
pi_parallel = zeros(1, n_rep_timing);
if use_parfor
    tic;
    parfor r = 1 : n_rep_timing
        xp = 2 * rand(N_timing, 1) - 1;
        yp = 2 * rand(N_timing, 1) - 1;
        pi_parallel(r) = 4 * sum(xp.^2 + yp.^2 <= 1) / N_timing;
    end
    t_parallel = toc;
else
    % Fallback: ruleaza serial si raporteaza acelasi timp
    tic;
    for r = 1 : n_rep_timing
        xp = 2 * rand(N_timing, 1) - 1;
        yp = 2 * rand(N_timing, 1) - 1;
        pi_parallel(r) = 4 * sum(xp.^2 + yp.^2 <= 1) / N_timing;
    end
    t_parallel = toc;
end

% Calculeaza speedup-ul (cat de mult e mai rapid parfor fata de for)
speedup = t_serial / t_parallel;

fprintf('Serial  (for):    %.3f secunde\n', t_serial);
if use_parfor
    fprintf('Paralel (parfor): %.3f secunde\n', t_parallel);
else
    fprintf('Paralel (parfor): %.3f secunde  [Toolbox indisponibil - for secvential]\n', ...
            t_parallel);
end
fprintf('Speedup: %.2fx\n', speedup);

%% SECTIUNEA 4: EROAREA vs N (scara log-log)
% -------------------------------------------------------
% Demonstram empiric convergenta Monte Carlo: eroarea absoluta |π_est - π|
% scade proportional cu 1/sqrt(N), conform Teoremei Limita Centrala.
% Pe scara log-log, aceasta relatie apare ca o dreapta cu panta -1/2.

% Linia teoretica 1/sqrt(N) scalata la eroarea primei valori din N_vec
% (scalare pentru a suprapune vizual cu datele empirice)
theoretical = pi_err(1) * sqrt(N_vec(1)) ./ sqrt(N_vec);

figure('Name', 'Eroarea Monte Carlo vs N', 'NumberTitle', 'off', ...
       'Position', [100 650 700 450]);

loglog(N_vec, pi_err,       'bo-', 'LineWidth', 2, 'MarkerSize', 7);
hold on;
loglog(N_vec, theoretical,  'r--', 'LineWidth', 1.5);

xlabel('N  [numarul de puncte]');
ylabel('Eroare absoluta  |π_{est} − π_{exact}|');
title('Convergenta erorilor Monte Carlo: eroarea ∝ 1/\surd{}N (scara log-log)');
legend({'Eroare empirica |π_{est} - π|', 'Referinta teoretica 1/\surd{}N'}, ...
       'Location', 'SouthWest');
grid on;
hold off;

%% CONCLUZII
% -------------------------------------------------------
% 1. Metoda Monte Carlo estimeaza π cu o eroare care scade ca 1/sqrt(N).
%    Pentru a obtine inca o cifra zecimala corecta, trebuie de 100 de ori
%    mai multe puncte → metoda este lenta, dar simpla si generala.
%
% 2. Operatiile logice (<=, ~, sum pe vector logic, indexare logica) sunt
%    esentiale pentru implementarea eficienta in MATLAB/Octave fara bucle
%    explicite (vectorizare).
%
% 3. parfor parallelizeaza eficient repetarile independente, obtinand un
%    speedup aproape liniar cu numarul de nuclee disponibile (overhead mic).
%
% 4. Aceasta metoda este un exemplu de INTEGRARE NUMERICA MONTE CARLO (C4):
%    π/4 = ∫∫_{[-1,1]^2} 1_{x^2+y^2 ≤ 1} dx dy
%    Spre deosebire de metodele clasice (Gauss, trapez), complexitatea
%    Monte Carlo nu creste exponential cu dimensiunea spatiului de integrare
%    → avantaj decisiv in dimensiuni inalte (fizica statistica, finante etc.)
%
% 5. Legea Numerelor Mari garanteaza convergenta estimatorului catre valoarea
%    adevarata pe masura ce N → ∞, independent de distributia punctelor.
% -------------------------------------------------------
fprintf('\nScript finalizat. Graficele sunt afisate in figuri.\n');
