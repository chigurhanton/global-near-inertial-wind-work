clear; clc;
%% =========================
%  ORAS5 MLD monthly files (curvilinear grid)
%  file pattern:
%  somxl030_control_monthly_highres_2D_YYYYMM_CONS_v0.1.nc
%% =========================
oras_dir = 'H:\oras5_mld_1980_2025\';   % ★ Change to your ORAS5 MLD directory
%% =========================
%  ERA5 monthly files list (sorted)
%% =========================
datadir2  = 'G:\era5_wind_speed_1940to2025\';   % ★ Change to your ERA5 monthly files directory
filelist2 = dir(fullfile(datadir2,'*.nc'));
[~, idxsort] = sort({filelist2.name});
filelist2 = filelist2(idxsort);
totalFiles = numel(filelist2);
%% =========================
%  constants
%% =========================
dt    = 3600;                           % fixed
rho0  = 1024;
omg_e = 2*pi/(23*3600 + 56*60 + 4);     % Earth's rotation rate (rad/s)
r     = 1/(3.7*24*3600);                % damping (s^-1)
%% =========================
%  carry-over state across months
%% =========================
Zi_prev = [];   % complex inertial current at previous month LAST hour: [row x col]
tt_prev = [];   % complex stress/rho at previous month LAST hour: [row x col]
% carry-over dynamics of previous month for cross-month step
A_prev = [];
B_prev = [];
omg_prev = [];
mask_prev = [];
is_first_month = true;
%% =========================
%  ORAS5 interpolant cache (curvilinear -> ERA5 regular grid)
%% =========================
F_oras = [];           % scatteredInterpolant
oras_cache = [];
oras_lon_src = [];     % [Nsrc,1]
oras_lat_src = [];     % [Nsrc,1]
oras_good_src = [];    % logical mask in flattened nav grid (same length as nav_lon(:))
%% =========================
%  loop
%% =========================
for k = 1:totalFiles
    filename2 = fullfile(datadir2, filelist2(k).name);
    percent = round((k / max(totalFiles,1)) * 100);
    fprintf('Processing file: %s ... %d%% completed\n', filelist2(k).name, percent);
    %% ---- read time ----
    info = ncinfo(filename2);
    varnames = {info.Variables.Name};
    if ismember('time', varnames)
        time1 = ncread(filename2, 'time');
        time  = datenum('1900-01-01 00:00:00') + double(time1)/24.0;
    elseif ismember('valid_time', varnames)
        time1 = ncread(filename2, 'valid_time');
        time  = datenum('1970-01-01 00:00:00') + double(time1)/86400;
    else
        error('Time variable not found. Existing variables include: \n%s', strjoin(varnames, ', '));
    end
    %% ---- read fields ----
    uu10 = ncread(filename2, 'u10');        % [row x col x nt]
    vv10 = ncread(filename2, 'v10');
    lon2 = ncread(filename2, 'longitude');  % length=row
    lat2 = ncread(filename2, 'latitude');   % length=col
    [row, col, nt] = size(uu10);
    %% ---- if lat descending, flip ----
    if numel(lat2) >= 2 && (lat2(2) < lat2(1))
        lat2 = flipud(lat2);
        uu10 = uu10(:, end:-1:1, :);
        vv10 = vv10(:, end:-1:1, :);
    end
    %% ---- parse YYYYMM from ERA5 filename ----
    tok6 = regexp(filelist2(k).name, '(\d{6})', 'tokens', 'once');
    if isempty(tok6)
        % Fallback: try parsing from the end of the path
        tok6 = regexp(filename2, '(\d{6})', 'tokens', 'once');
    end
    if isempty(tok6)
        error('Cannot parse YYYYMM from ERA5 filename: %s', filelist2(k).name);
    end
    name_yy_mm = str2double(tok6{1});
    yy = floor(name_yy_mm/100);
    mm = mod(name_yy_mm, 100);
    if isnan(yy) || isnan(mm) || mm<1 || mm>12
        error('Parsed invalid year/month: %d (yy=%d,mm=%d) from %s', name_yy_mm, yy, mm, filelist2(k).name);
    end
    %% ---- init carry-over arrays ----
    if is_first_month
        Zi_prev = complex(zeros(row, col), zeros(row, col));
        tt_prev = complex(zeros(row, col), zeros(row, col));
    else
        if ~isequal(size(Zi_prev), [row, col]) || ~isequal(size(tt_prev), [row, col])
            warning('Grid size changed. Resetting cross-month state to 0 (will introduce a discontinuity). File: %s', filelist2(k).name);
            Zi_prev = complex(zeros(row, col), zeros(row, col));
            tt_prev = complex(zeros(row, col), zeros(row, col));
            A_prev = []; B_prev = []; omg_prev = []; mask_prev = [];
        end
    end
    %% =========================
    % Step 1: wind stress for whole field (vectorized)
    %% =========================
    [tx3, ty3] = computate_wind_stress_1218(uu10, vv10);  % [row col nt]
    tt3 = (tx3 + 1i*ty3) ./ rho0;                         % complex stress/rho
    %% =========================
    % Step 2: ORAS5 MLD -> ERA5 grid (curvilinear -> regular)
    [H, mask_H] = interp_oras5_mld_to_era5(oras_dir, yy, mm, lon2, lat2);
    %% =========================
    % Step 3: dynamics fields for THIS month
    %% =========================
    [~, LatFull] = ndgrid(lon2, lat2);               % LatFull [row col]
    f2   = 2*omg_e*sind(LatFull);
    omg2 = (r + 1i*f2);
    mask_lat = (abs(LatFull) > 3) & (abs(LatFull) <= 60);
    mask_valid = mask_H & mask_lat;
    A = exp(-omg2 * dt);
    B = -(1 - A) ./ (H .* (omg2.^2));
    A(~mask_valid) = NaN;
    B(~mask_valid) = NaN;
    omg2(~mask_valid) = NaN;
    %% =========================
    % Step 4: allocate outputs & online ICF stats
    %% =========================
    u_i = NaN(row, col, nt, 'single');
    v_i = NaN(row, col, nt, 'single');
    ICF_sum  = zeros(row, col, 'double');
    ICF_cnt  = zeros(row, col, 'double');
    WORK_sum = zeros(row, col, 'double');
    %% =========================
    % Step 5: Zi at THIS month first hour + ICF at first hour
    %% =========================
    if is_first_month || isempty(A_prev)
        Zi = complex(zeros(row, col), zeros(row, col));
        Zi(~mask_valid) = NaN;
        ICF_1 = NaN(row, col);
    else
        T_cross = (tt3(:,:,1) - tt_prev) / dt;
        Zi = Zi_prev .* A_prev + B_prev .* T_cross;
        mask_cross = mask_prev & mask_valid;
        Zi(~mask_cross) = NaN;
        ICF_1 = -rho0 * real( T_cross .* conj(Zi) ./ omg_prev );
        ICF_1(~mask_cross) = NaN;
    end
    u_i(:,:,1) = single(real(Zi));
    v_i(:,:,1) = single(imag(Zi));
    % online accumulate first-hour ICF
    m1 = isfinite(ICF_1);
    ICF_sum(m1)  = ICF_sum(m1)  + ICF_1(m1);
    ICF_cnt(m1)  = ICF_cnt(m1)  + 1;
    WORK_sum(m1) = WORK_sum(m1) + ICF_1(m1) * dt;
    %% =========================
    % Step 6: within-month stepping (THIS month dynamics)
    %% =========================
    for n = 1:(nt-1)
        Tn = (tt3(:,:,n+1) - tt3(:,:,n)) / dt;
        Zi = A .* Zi + B .* Tn;
        Zi(~mask_valid) = complex(NaN, NaN);
        u_i(:,:,n+1) = single(real(Zi));
        v_i(:,:,n+1) = single(imag(Zi));
        ICF_now = -rho0 * real( Tn .* conj(Zi) ./ omg2 );
        ICF_now(~mask_valid) = NaN;
        m = isfinite(ICF_now);
        ICF_sum(m)  = ICF_sum(m)  + ICF_now(m);
        ICF_cnt(m)  = ICF_cnt(m)  + 1;
        WORK_sum(m) = WORK_sum(m) + ICF_now(m) * dt;
    end
    %% =========================
    % Step 7: update carry-over for next month
    %% =========================
    Zi_prev = Zi;
    tt_prev = tt3(:,:,end);
    A_prev    = A;
    B_prev    = B;
    omg_prev  = omg2;
    mask_prev = mask_valid;
    %% =========================
    % outputs
    %% =========================
    outfile = fullfile('H:\inertial_with_mld', sprintf('inertial_%d.nc', name_yy_mm));   % ★ Change output directory/drive
    write_inertial_nc(outfile, lon2, lat2, time, u_i, v_i);
    % ICF_m = ICF_sum ./ max(ICF_cnt, 1);
    % ICF_m(ICF_cnt == 0) = NaN;
    % 
    % WORK_m = WORK_sum;
    % year_month = name_yy_mm; %#ok<NASGU>
    % If you need to save ICF/WORK mat files, uncomment and change the path:
    % save(fullfile('G:\...\ICF', sprintf('ICF_%d.mat', name_yy_mm)), 'ICF_m');
    % save(fullfile('G:\...\WORK', sprintf('WORK_%d.mat', name_yy_mm)), 'WORK_m');
    is_first_month = false;
end