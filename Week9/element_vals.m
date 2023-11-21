% actual values
clear all;
wc = (1e-2)*1e9; % 5 GHz
R0 = 50;
L0 = 1e-7; %[nH/m]


% ----------- Line parameters for prototype low pass filter --------------

N = 16; L0 = L0; Rs = R0; RL = R0;

% ----------- Test Matrices for w and C ----------
w = linspace(100, 1e9, 100);
C = transpose(linspace(4e-21, 4e-20, 3));

C_sel = zeros(N, 2);
Zin_sec = zeros(N+1, length(w));
Plr_sec = zeros(N, length(w));

% ----------- Comparison matrix -------------
Plr_req = 1 + (w./wc).^4;
Plr_sec(1, : ) = Plr_req;

% ----------- Start iteration -------------------

Zin_prev = ones(1, length(w));
Zin_sel(1, : ) = Zin_prev;

figure; title('P_{lr}');
rows = length(C);


% Create a square grid of subplots
gridSize = ceil(sqrt(N));

for i=1:1:N
    Zin_curr = 2j.*L0.*w + (Zin_prev)./(1 + 1j.*(w.*Zin_prev).*C);
    disp(Zin_curr);
    Plr = 1 + (1./(4.*Zin_prev)).* ( (1-Zin_prev).^2 + (Zin_prev.^2.*C.^2 + L0.^2 - 2.*L0.*C.*Zin_prev).*w.^2 + (4.*L0.^2.*C.^2.*Zin_prev.^2).*w.^4 );
    %Plr = (Zin_curr+1).*conj(Zin_curr+1)./(2.*(Zin_curr + conj(Zin_curr)));

    plot_Plrs(Plr, Plr_req, w, C, i, gridSize);
    plot(w, log(Plr_req), 'x');
    hold off;


    [min_disparity, optimal_row] = select_best_C(Plr, Plr_req);
    C_sel(i, 1) = C(optimal_row);
    C_sel(i, 2) = min_disparity;

    Zin_curr = Zin_curr(optimal_row, : ); % choose best result
    Zin_sec(i+1, : ) = Zin_curr;
    Plr_sec(i+1, : ) = abs(Plr(optimal_row, : ));

    Zin_prev = Zin_curr;

end


plot_together_log(w, Zin_sec, gridSize, N, 'Z_{in}');
plot_together(w, Plr_sec, gridSize, N, 'P_{lr}');

% plot the Plr for different C values
function plot_Plrs(Plr, Plr_req, w, C, i, gridSize)
    subplot(gridSize, gridSize, i);
    rows = length(Plr(:, 1));
    
    ylabel('w');
    for j=1:1:rows
        plot(w, log(abs(Plr(j, :))));
        hold on;
    end
end

function plot_together(w, y, gridSize, N, title_str)
    figure;
    for i=1:1:N
        subplot(gridSize, gridSize, i);
        plot(w, abs(y(i, :)));
        hold on;
    end
    title(title_str);
    hold off;
end

function plot_together_log(w, y, gridSize, N, title_str)
    figure;
    for i=1:1:N
        subplot(gridSize, gridSize, i);
        plot(w, 20*log(abs(y(i, :))));
        hold on;
    end
    title(title_str);
    hold off;
end


function [min_avg, optimal_row] =  select_best_C(Plr, Plr_req)
    Plr_diff = abs(Plr - Plr_req);
    row_avg = mean(Plr_diff, 2); % Calculate the average of each row
    [min_avg, optimal_row] = min(row_avg); % Find the minimum average value and its corresponding row number
end