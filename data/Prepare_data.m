clear
clc
fn = 'F:\Matlab_Code_For_Global_Wind_Work_for1940_1978\code_for_drifter\gdp_hourly.nc';
rowsize = ncread(fn, 'rowsize');   % [traj]
ID      = ncread(fn, 'ID');        % [traj]
lon_all = double(ncread(fn,'lon'));  % [obs]
lon_all(lon_all < 0) = lon_all(lon_all < 0) + 360;   % -> [0,360)
% Optional: prevent 360 from appearing (theoretically won't happen, but just to be safe)
% figure
% plot(lon_all)
lon_all(lon_all >= 360) = lon_all(lon_all >= 360) - 360;
lat_all = ncread(fn, 'lat');       % [obs]
ve_all  = ncread(fn, 've');        % [obs]
vn_all  = ncread(fn, 'vn');        % [obs]
time_all= ncread(fn, 'time');      % [obs] seconds since 1970-01-01
drogue  = ncread(fn, 'drogue_status'); % [obs] 1 drogued, 0 undrogued
gap_all = ncread(fn, 'gap');       % [obs]

% Convert to Julian days (days since 1970-01-01) - numeric format, fast computation
time_all_days = time_all / 86400;  % Convert to days

% ========= Trajectory indices =========
ntraj   = length(rowsize);
cumrows = cumsum(rowsize);            % Cumulative number of observations
idx_start = [1; cumrows(1:end-1)+1];  % Starting obs index for each trajectory
idx_end   = cumrows;                  % Ending obs index for each trajectory
Nobs = numel(time_all);

% ========= Sliding window parameters =========
seg_len = 300;   % 300 hours
step    = 150;   % Half overlap (150 hours)

% ========= Collect each 300h segment (raw current) =========
u_raw_seg_list   = {};   % Each cell: [300x1] raw U
v_raw_seg_list   = {};   % Each cell: [300x1] raw V
lon_seg_list  = {};   % Each cell: [300x1] longitude
lat_seg_list  = {};   % Each cell: [300x1] latitude
f_seg_list    = {};
time_seg_list = {};   % Each cell: [300x1] time (days)
trajID_list   = [];   % Drifter ID corresponding to each segment
seg_start_idx_list = []; % Starting index within the trajectory (optional)

for k = 1:ntraj
    i1  = idx_start(k);
    i2  = idx_end(k);
    ind = (i1:i2)';
    lon_k    = lon_all(ind);
    lat_k    = lat_all(ind);
    ve_k_raw = double(ve_all(ind));
    vn_k_raw = double(vn_all(ind));
    t_k      = time_all_days(ind);  % Use days (numeric format)
    gap_k    = gap_all(ind);
    drogue_k = drogue(ind);
    
    % Skip if the entire trajectory is too short
    Nk = numel(t_k);
    if Nk < seg_len
        continue
    end
    
    % ========= Apply 300h window on this trajectory =========
    for s_start = 1:step:(Nk - seg_len + 1)
        s_end   = s_start + seg_len - 1;
        seg_idx = s_start:s_end;          % Index within the trajectory
        
        % Extract raw data for this segment
        ve_seg_raw = ve_k_raw(seg_idx);
        vn_seg_raw = vn_k_raw(seg_idx);
        lat_seg    = lat_k(seg_idx);
        lon_seg    = lon_k(seg_idx);        
        gap_seg    = gap_k(seg_idx);
        drogue_seg = drogue_k(seg_idx);
        t_seg      = t_k(seg_idx);  % Days (numeric format)
        
        % ========== 1) Segment-level QC: drogue ratio & gap ==========
        nonnan_seg = ~isnan(ve_seg_raw) & ~isnan(vn_seg_raw);
        if sum(nonnan_seg) < 0.8 * seg_len
            continue
        end
        drogue_mask = (drogue_seg == 1);
        seg_frac_drogued = sum(drogue_mask & nonnan_seg) / sum(nonnan_seg);
        if seg_frac_drogued < 0.5
            % Most of the time in this segment is not drogued, skip analysis
            continue
        end
        gap_max = max(gap_seg(nonnan_seg), [], 'omitnan');
        if gap_max > 10 * 3600
            continue
        end
        
        % ========== 2) Inertial frequency (take the average of valid latitudes in the segment) ==========
        valid_for_phi = drogue_mask & nonnan_seg;
        phi = mean(lat_seg(valid_for_phi), 'omitnan');
        if isnan(phi) || abs(phi) < 3 || abs(phi) > 60
            % Exclude equator ±3° and |lat|>60° for now
            continue
        end
   
        % ========== 6) Store ==========
        u_raw_seg_list{end+1}   = ve_seg_raw(:);   % raw U, 300x1
        v_raw_seg_list{end+1}   = vn_seg_raw(:);   % raw V, 300x1
    end
end

%% ========= Convert cell array to [300 x Nseg] matrix for unified saving =========
Nseg = numel(v_raw_seg_list);
fprintf('Total %d 300h segments (samples) obtained\n', Nseg);
seg_len = 300;
u_raw_300   = nan(seg_len, Nseg);
v_raw_300   = nan(seg_len, Nseg);
for n = 1:Nseg
    u_raw_300(:,n) = u_raw_seg_list{n}(:);
    v_raw_300(:,n) = v_raw_seg_list{n}(:);
end

%% Save data
out_fn = 'gdp_drifter_raw_uv_300h_segments.mat';
save(out_fn, ...
    'u_raw_300','v_raw_300', ...
    '-v7.3');
fprintf('Saved 300h raw current segment data to %s\n', out_fn);




%%
clear
clc
fn = 'F:\Matlab_Code_For_Global_Wind_Work_for1940_1978\code_for_drifter\gdp_hourly.nc';
rowsize = ncread(fn, 'rowsize');   % [traj]
ID      = ncread(fn, 'ID');        % [traj]
lon_all = double(ncread(fn,'lon'));  % [obs]
lon_all(lon_all < 0) = lon_all(lon_all < 0) + 360;   % -> [0,360)
% Optional: prevent 360 from appearing (theoretically won't happen, but just to be safe)
% figure
% plot(lon_all)
lon_all(lon_all >= 360) = lon_all(lon_all >= 360) - 360;
lat_all = ncread(fn, 'lat');       % [obs]
ve_all  = ncread(fn, 've');        % [obs]
vn_all  = ncread(fn, 'vn');        % [obs]
time_all= ncread(fn, 'time');      % [obs] seconds since 1970-01-01
drogue  = ncread(fn, 'drogue_status'); % [obs] 1 drogued, 0 undrogued
gap_all = ncread(fn, 'gap');       % [obs]
% Convert to Julian days (days since 1970-01-01) - numeric format, fast computation
time_all_days = time_all / 86400;  % Convert to days
% ========= Trajectory indices =========
ntraj   = length(rowsize);
cumrows = cumsum(rowsize);            % Cumulative number of observations
idx_start = [1; cumrows(1:end-1)+1];  % Starting obs index for each trajectory
idx_end   = cumrows;                  % Ending obs index for each trajectory
Nobs = numel(time_all);
% ========= Sliding window parameters =========
seg_len = 300;   % 300 hours
step    = 150;   % Half overlap (150 hours)
% ========= Collect each 300h segment (ground truth inertial current) =========
uI_seg_list   = {};   % Each cell: [300x1] inertial U
vI_seg_list   = {};   % Each cell: [300x1] inertial V
lon_seg_list  = {};   % Each cell: [300x1] longitude
lat_seg_list  = {};   % Each cell: [300x1] latitude
f_seg_list    = {};
time_seg_list = {};   % Each cell: [300x1] time (days)
trajID_list   = [];   % Drifter ID corresponding to each segment
seg_start_idx_list = []; % Starting index within the trajectory (optional)

for k = 1:ntraj
    i1  = idx_start(k);
    i2  = idx_end(k);
    ind = (i1:i2)';
    lon_k    = lon_all(ind);
    lat_k    = lat_all(ind);
    ve_k_raw = double(ve_all(ind));
    vn_k_raw = double(vn_all(ind));
    t_k      = time_all_days(ind);  % Use days (numeric format)
    gap_k    = gap_all(ind);
    drogue_k = drogue(ind);
    
    % Skip if the entire trajectory is too short
    Nk = numel(t_k);
    if Nk < seg_len
        continue
    end
    
    % ========= Apply 300h window on this trajectory =========
    for s_start = 1:step:(Nk - seg_len + 1)
        s_end   = s_start + seg_len - 1;
        seg_idx = s_start:s_end;          % Index within the trajectory
        
        % Extract raw data for this segment
        ve_seg_raw = ve_k_raw(seg_idx);
        vn_seg_raw = vn_k_raw(seg_idx);
        lat_seg    = lat_k(seg_idx);
        lon_seg    = lon_k(seg_idx);        
        gap_seg    = gap_k(seg_idx);
        drogue_seg = drogue_k(seg_idx);
        t_seg      = t_k(seg_idx);  % Days (numeric format)
        
        % ========== 1) Segment-level QC: drogue ratio & gap ==========
        nonnan_seg = ~isnan(ve_seg_raw) & ~isnan(vn_seg_raw);
        if sum(nonnan_seg) < 0.8 * seg_len
            continue
        end
        drogue_mask = (drogue_seg == 1);
        seg_frac_drogued = sum(drogue_mask & nonnan_seg) / sum(nonnan_seg);
        if seg_frac_drogued < 0.5
            % Most of the time in this segment is not drogued, skip inertial analysis
            continue
        end
        gap_max = max(gap_seg(nonnan_seg), [], 'omitnan');
        if gap_max > 10 * 3600
            continue
        end
        
        % ========== 2) Inertial frequency (take the average of valid latitudes in the segment) ==========
        valid_for_phi = drogue_mask & nonnan_seg;
        phi = mean(lat_seg(valid_for_phi), 'omitnan');
        if isnan(phi) || abs(phi) < 3 || abs(phi) > 60
            % Exclude equator ±3° and |lat|>60° for now
            continue
        end
        omega_earth = 1 / (23 * 3600 + 56 * 60 + 4);   % rad/s
        f_inertial  = abs(2 * omega_earth * sind(phi));  % s^-1
        f_inertial_true  = 2 * omega_earth * sind(phi);  % s^-1
        lon0 = lon_seg(1);
        lat0 = phi;
        
        % ========== 3) Inertial bandpass filter ==========
        rp = 3;
        rs = 40;
        Fs = 1/3600;              % Sampling frequency (Hz)
        Fa = Fs/2;                % Nyquist
        wp = [0.75 1.25] * f_inertial;   % Passband
        ws = [0.7  1.3 ] * f_inertial;   % Stopband
        if wp(1) <= 0 || wp(2) >= Fa
            continue
        end
        [~, Wn] = buttord(wp/Fa, ws/Fa, rp, rs);
        [b_f, a_f] = butter(2, Wn, 'bandpass');
        
        % ========== 4) Fill small gaps (<=6h) to prevent filtfilt from crashing ==========
        ve_seg_d = fillmissing(ve_seg_raw, 'linear', 'MaxGap', 6);
        vn_seg_d = fillmissing(vn_seg_raw, 'linear', 'MaxGap', 6);
        if any(isnan(ve_seg_d) | isnan(vn_seg_d))
            continue
        end
        
        % ========== 5) Inertial bandpass filtering (filter these 300h together) ==========
        ve_I_seg = filtfilt(b_f, a_f, ve_seg_d);
        vn_I_seg = filtfilt(b_f, a_f, vn_seg_d);
        
        % ========== 6) Store ==========
        uI_seg_list{end+1}   = ve_I_seg(:);   % Inertial U, 300x1
        vI_seg_list{end+1}   = vn_I_seg(:);   % Inertial V, 300x1
        lon_seg_list{end+1}  = lon0 * ones(seg_len,1);
        lat_seg_list{end+1}  = lat0 * ones(seg_len,1);
        f_seg_list{end+1}    = f_inertial_true * ones(seg_len,1);
        time_seg_list{end+1} = t_seg(:);      % Time, days (numeric format)
        trajID_list(end+1,1)       = ID(k);       % Which drifter this segment belongs to
        seg_start_idx_list(end+1,1)= seg_idx(1);  % Starting index in the trajectory
    end
end

%% ========= Convert cell array to [300 x Nseg] matrix for unified saving =========
Nseg = numel(uI_seg_list);
fprintf('Total %d 300h segments (samples) obtained\n', Nseg);
seg_len = 300;
uI_300   = nan(seg_len, Nseg);
vI_300   = nan(seg_len, Nseg);
lon_300  = nan(seg_len, Nseg);
lat_300  = nan(seg_len, Nseg);
f_300    = nan(seg_len, Nseg);
time_300 = nan(seg_len, Nseg);  % Numeric matrix, no time zone issues
for n = 1:Nseg
    uI_300(:,n)   = uI_seg_list{n}(:);
    vI_300(:,n)   = vI_seg_list{n}(:);
    lon_300(:,n)  = lon_seg_list{n}(:);
    lat_300(:,n)  = lat_seg_list{n}(:);
    f_300(:,n)    = f_seg_list{n}(:);
    time_300(:,n) = time_seg_list{n}(:);  % Direct assignment, extremely fast
end

%% Save data
out_fn = 'gdp_drifter_NIW_300h_segments.mat';
save(out_fn, ...
    'uI_300','vI_300','lon_300','lat_300','time_300','f_300', ...
    '-v7.3');
fprintf('Saved 300h inertial current segment data to %s\n', out_fn);





%%
clear
S = load('gdp_drifter_NIW_300h_segments.mat', ...
         'lon_300','lat_300','time_300');
lon_300  = S.lon_300;    % [T, Nseg]
lat_300  = S.lat_300;    % [T, Nseg]
time_300 = S.time_300;   % [T, Nseg]

datetime_300 = datetime(1970,1,1,0,0,0) + days(time_300);
[T, Nseg] = size(time_300);   % T should be 300

% 2) Pre-allocate for output variables (placeholder with NaN, fill month-by-month later)
u_i_300 = nan(T, Nseg);
v_i_300 = nan(T, Nseg);

% 3) Calculate year and month for all time points
year_mat  = year(datetime_300);    % [T, Nseg]
month_mat = month(datetime_300);   % [T, Nseg]

ym_all = year_mat*100 + month_mat;  % e.g., 201201
ym_list = unique(ym_all);      % All unique year-months

era_dir = 'H:\inertial_with_mld\';

% 4) Outer loop by (year, month)
for kym = 1:numel(ym_list)
    ym = ym_list(kym);
    yy = floor(ym/100);
    mm = mod(ym,100);
    
    era_fn = sprintf('%sinertial_%4d%02d.nc', era_dir, yy, mm);
    if ~isfile(era_fn)
        fprintf('No inertial file: %s, skip this month.\n', era_fn);
        continue
    end
    
    fprintf('Processing (%s) at %s\n', era_fn, datestr(now, 'HH:MM:SS'));
    
    % === 4.1 Read ERA5 data for this month only once ===
    lon_e  = ncread(era_fn,'longitude');   % [nlon]
    lat_e  = ncread(era_fn,'latitude');    % [nlat]
    
    time_1 = ncread(era_fn, 'time');
    time_e_dt = datetime(time_1, 'ConvertFrom', 'datenum');
    time_e_hr = dateshift(time_e_dt,'start','hour'); % Align to the start of the hour
    
    u_i = ncread(era_fn,'u_i');   % [nlon x nlat x ntime]
    v_i = ncread(era_fn,'v_i');
    
    % === 4.2 Find all points in time_300 belonging to this (yy,mm) ===
    mask_month = (year_mat == yy) & (month_mat == mm);  % [T,Nseg]
    if ~any(mask_month(:))
        continue
    end
    
    % Process time points within this month for each segment
    for n = 1:Nseg
        idx_t_seg = find(mask_month(:,n));   % Time indices in this segment belonging to this month
        if isempty(idx_t_seg), continue; end
        
        % Fixed point for this segment: lon/lat are constant within the segment, just take the first point
        lon0 = lon_300(1,n);
        lat0 = lat_300(1,n);
        
        % ——★ Spatially find the nearest grid point first (calculate only once) ★——
        [~, ix] = min(abs(lon_e - lon0));   % Nearest longitude grid point
        [~, iy] = min(abs(lat_e - lat0));   % Nearest latitude grid point
        
        % Extract these time points and align to the start of the hour
        t_seg_sub    = datetime_300(idx_t_seg, n);
        t_seg_sub_hr = dateshift(t_seg_sub, 'start', 'hour');
        
        % Find corresponding moments on the ERA5 time axis
        [~, idx_e] = ismember(t_seg_sub_hr, time_e_hr);
        
        valid_idx  = idx_e > 0;
        if ~any(valid_idx), continue; end
        
        idx_t_seg_valid = idx_t_seg(valid_idx);
        idx_e_valid     = idx_e(valid_idx);
        
        % ——★ Directly use nearest grid point & vector assignment (no longer interpolate hour by hour) ★——
        % u_i(ix,iy,:) is [1x1xntime], squeeze the first two dimensions to get [ntime_valid x 1]
        u_vals = squeeze(u_i(ix, iy, idx_e_valid));   % Column vector
        v_vals = squeeze(v_i(ix, iy, idx_e_valid));
        
        u_i_300(idx_t_seg_valid, n) = u_vals;
        v_i_300(idx_t_seg_valid, n) = v_vals;
    end
end

% 5) Save results
save('drifter_inertial_u_iv_i_fixedpoint_300h_20260112_fororas5.mat', ...
     'u_i_300','v_i_300','-v7.3');



%% build_H300_from_oras5_monthly_mat_large.m
% Suitable for: Nseg~4e5, T=300
% Features: row-by-row (t) processing, avoids creating large yyyymm_mat/mask, saves memory
clear; clc;

% ========= 0) Paths =========
seg_mat  = 'gdp_drifter_NIW_300h_segments.mat';
oras_dir = 'H:\oras5_mld_1980_2022\mld_monthly\';  % Directory containing oras5_mld_YYYYMM.mat files
out_mat  = 'drifter_oras5_MLD_300h.mat';

% ========= 1) Read segments =========
S = load(seg_mat, 'lon_300','lat_300','time_300');
lon_300  = S.lon_300;      % [T, Nseg]
lat_300  = S.lat_300;      % [T, Nseg]
time_300 = S.time_300;     % double(days since 1970) or datetime
[T, Nseg] = size(lon_300);
fprintf('T=%d, Nseg=%d\n', T, Nseg);

% ========= 2) Fixed positions (use only the first row) =========
lon0 = mod(lon_300(1,:), 360);   % [1, Nseg]
lat0 = lat_300(1,:);

% ========= 3) Regular grid definition (matches your saved mat) =========
resolution = 0.25;
lon_min = 0;
lat_min = -80;
nlon = 1440;               % 0:0.25:359.75
nlat = 681;                % -80:0.25:90

% Lon/lat -> nearest grid point index (lat, lon)
ix = round((lon0 - lon_min)/resolution) + 1;   % 1..1440
iy = round((lat0 - lat_min)/resolution) + 1;   % 1..681

% Out-of-bounds handling: it is more rigorous to set directly to NaN (rather than clamping)
bad = (ix<1)|(ix>nlon)|(iy<1)|(iy>nlat);
ix(bad) = 1; iy(bad) = 1;    % Set to a valid value temporarily, mask with 'bad' later
ix = uint16(ix); iy = uint16(iy);

% ========= 4) Outputs =========
H_300 = nan(T, Nseg, 'single');
if any(bad)
    fprintf('Warning: %d segments out of [-80,90]x[0,360), will stay NaN.\n', sum(bad));
end

% ========= 5) MAT caching: prevents repeatedly loading the same month (very important) =========
cache_ym  = uint32(0);
cache_mld = [];   % single [nlat x nlon]

% ========= 6) Main loop: process row-by-row (t) =========
% Since time_300 is days since 1970, convert to datetime row-by-row here to avoid generating a massive datetime matrix
base_dt = datetime(1970,1,1,0,0,0);
for t = 1:T
    % ---- Calculate yyyymm for all segments in this row (length Nseg) ----
    if isdatetime(time_300)
        dt_row = time_300(t,:);              % datetime 1xNseg
    else
        dt_row = base_dt + days(time_300(t,:));  % 1xNseg datetime
    end
    ym_row = uint32(year(dt_row)*100 + month(dt_row));  % 1xNseg
    
    % ---- Group by yyyymm (only done for this row) ----
    ym_u = unique(ym_row);    % One row usually contains only a few months (cross-month segments have at most 2-3)
    
    for k = 1:numel(ym_u)
        ym = ym_u(k);
        idx = find(ym_row == ym);          % Segments in this row belonging to this month
        idx = idx(~bad(idx));              % Remove out-of-bounds segments (keep as NaN)
        if isempty(idx), continue; end
        
        % ---- Load / cache the MLD grid for this month ----
        if cache_ym ~= ym
            matfile = fullfile(oras_dir, sprintf('oras5_mld_%d.mat', ym));
            if ~isfile(matfile)
                % Missing month: keep as NaN
                warning('Missing MAT: %s (t=%d)', matfile, t);
                cache_ym = uint32(0);
                cache_mld = [];
                continue
            end
            M = load(matfile, 'mld_single_month');
            mld = M.mld_single_month;
            if ~isa(mld,'single'); mld = single(mld); end
            cache_ym  = ym;
            cache_mld = mld;
            clear M mld
        end
        
        % ---- Extract values at once using linear indices (lat, lon order!) ----
        lin = sub2ind([nlat, nlon], double(iy(idx)), double(ix(idx)));
        H_300(t, idx) = cache_mld(lin);
    end
    
    if mod(t,50)==0 || t==1 || t==T
        fprintf('Done t=%d/%d\n', t, T);
    end
end

% ========= 7) Save =========
save(out_mat, 'H_300', 'resolution', '-v7.3');
fprintf('Saved: %s\n', out_mat);



%%
clear
S = load('gdp_drifter_NIW_300h_segments.mat', ...
         'lon_300','lat_300','time_300');
lon_300  = S.lon_300;    % [T, Nseg]
lat_300  = S.lat_300;    % [T, Nseg]
time_300 = S.time_300;   % [T, Nseg]

datetime_300 = datetime(1970,1,1,0,0,0) + days(time_300);
[T, Nseg] = size(time_300);   % T should be 300

% Output pre-allocation
u10_300 = nan(T, Nseg);
v10_300 = nan(T, Nseg);
tx_300  = nan(T, Nseg);
ty_300  = nan(T, Nseg);

% ★ New: Wind stress difference (delta stress)
dtx_300 = nan(T, Nseg);
dty_300 = nan(T, Nseg);

% Year and month matrices
year_mat  = year(datetime_300);
month_mat = month(datetime_300);

ym_all  = year_mat*100 + month_mat;
ym_list = unique(ym_all);

era_dir = 'E:\era5_wind_data_1987to2024\';

for kym = 1:numel(ym_list)
    ym = ym_list(kym);
    yy = floor(ym/100);
    mm = mod(ym,100);
    
    era_fn = sprintf('%sera5_daily_%4d%02d.nc', era_dir, yy, mm);
    if ~isfile(era_fn)
        fprintf('No ERA5 file: %s, skip this month.\n', era_fn);
        continue
    end
    
    fprintf('Processing (%s) at %s\n', era_fn, datestr(now, 'HH:MM:SS'));
    
    % === 4.1 Read ERA5 for the current month (and possibly prepend the last hour of the previous month) ===
    lon_e  = ncread(era_fn,'longitude');   % [nlon]
    lat_e  = ncread(era_fn,'latitude');    % [nlat]
    
    info = ncinfo(era_fn);
    varnames = {info.Variables.Name};
    if ismember('time', varnames)
        time_1 = ncread(era_fn, 'time');
        time_e_dt = datetime(1900, 1, 1, 0, 0, 0) + hours(double(time_1));
    elseif ismember('valid_time', varnames)
        time_1 = ncread(era_fn, 'valid_time');
        time_e_dt = datetime(1970, 1, 1, 0, 0, 0) + seconds(double(time_1));
    else
        error('Time variable not found, existing variables include: \n%s', strjoin(varnames, ', '));
    end
    
    time_e_hr = dateshift(time_e_dt,'start','hour'); % Align to the start of the hour
    
    u10 = ncread(era_fn,'u10');   % [nlon x nlat x ntime]
    v10 = ncread(era_fn,'v10');
    [Tx, Ty] = computate_wind_stress_1218(u10, v10);
    
    % === Check if any segment in this month needs a "previous hour" that falls before the start of the month ===
    mask_month = (year_mat == yy) & (month_mat == mm);  % [T,Nseg]
    if ~any(mask_month(:))
        continue
    end
    
    % Find all required time points (hourly) in this month
    t_need_all = dateshift(datetime_300(mask_month), 'start', 'hour');
    t_need_min = min(t_need_all);
    t_prev_min = t_need_min - hours(1);
    
    % If the required previous hour is earlier than the first hour of the current month's file, prepend the last hour of the previous month
    if t_prev_min < time_e_hr(1)
        % Calculate previous month (yy_prev, mm_prev)
        yy_prev = yy; mm_prev = mm - 1;
        if mm_prev == 0
            mm_prev = 12; yy_prev = yy - 1;
        end
        era_fn_prev = sprintf('%sera5_daily_%4d%02d.nc', era_dir, yy_prev, mm_prev);
        
        if isfile(era_fn_prev)
            info2 = ncinfo(era_fn_prev);
            varnames2 = {info2.Variables.Name};
            
            % Read previous month's time to locate the last hour
            if ismember('time', varnames2)
                time_p = ncread(era_fn_prev, 'time');
                time_p_dt = datetime(1900, 1, 1, 0, 0, 0) + hours(double(time_p));
            elseif ismember('valid_time', varnames2)
                time_p = ncread(era_fn_prev, 'valid_time');
                time_p_dt = datetime(1970, 1, 1, 0, 0, 0) + seconds(double(time_p));
            else
                error('Time variable not found in the previous month''s file: %s', era_fn_prev);
            end
            time_p_hr = dateshift(time_p_dt,'start','hour');
            
            % Get the last time index of the previous month
            it_last = numel(time_p_hr);
            
            % Read only the last hour of u10/v10 (and corresponding stress)
            u10_last = ncread(era_fn_prev,'u10', [1 1 it_last], [Inf Inf 1]); % [nlon x nlat x 1]
            v10_last = ncread(era_fn_prev,'v10', [1 1 it_last], [Inf Inf 1]);
            [Tx_last, Ty_last] = computate_wind_stress_1218(u10_last, v10_last);
            
            % ★ Prepend to the beginning of the current month's arrays: time / u10 / v10 / Tx / Ty
            time_e_hr = [time_p_hr(it_last); time_e_hr];
            u10 = cat(3, u10_last, u10);
            v10 = cat(3, v10_last, v10);
            Tx  = cat(3, Tx_last,  Tx);
            Ty  = cat(3, Ty_last,  Ty);
            
            % figure
            % subplot(3, 1, 1)
            % s = pcolor(u10_last(:, :)');
            % s.FaceColor = "interp";
            % s.EdgeColor ="none";
            % colorbar
            % subplot(3, 1, 2)
            % s = pcolor(u10(:, :, 1)');
            % s.FaceColor = "interp";
            % s.EdgeColor ="none";
            % colorbar
            % subplot(3, 1, 3)
            % s = pcolor(u10(:, :, 1)'-u10_last(:, :)');
            % s.FaceColor = "interp";
            % s.EdgeColor ="none";
            % colorbar
            
            fprintf('  prepended prev-month last hour: %s\n', datestr(time_p_hr(it_last)));
        else
            % No previous month file: delta for those points at the beginning of the month will remain NaN
            fprintf('  WARNING: need prev-hour but no prev file: %s (delta at month-start will be NaN)\n', era_fn_prev);
        end
    end
    
    % === 4.2 Process the time points belonging to each segment within this month ===
    for n = 1:Nseg
        idx_t_seg = find(mask_month(:,n));
        if isempty(idx_t_seg), continue; end
        
        lon0 = lon_300(1,n);
        lat0 = lat_300(1,n);
        
        [~, ix] = min(abs(lon_e - lon0));
        [~, iy] = min(abs(lat_e - lat0));
        
        t_seg_sub    = datetime_300(idx_t_seg, n);
        t_seg_sub_hr = dateshift(t_seg_sub, 'start', 'hour');
        
        % Map to ERA5 time indices (time_e_hr might have been prepended by now)
        [~, idx_e] = ismember(t_seg_sub_hr, time_e_hr);
        
        valid_idx  = idx_e > 0;
        if ~any(valid_idx), continue; end
        
        idx_t_seg_valid = idx_t_seg(valid_idx);
        idx_e_valid     = idx_e(valid_idx);
        
        % Extract values
        u_vals  = squeeze(u10(ix, iy, idx_e_valid));
        v_vals  = squeeze(v10(ix, iy, idx_e_valid));
        tx_vals = squeeze(Tx(ix, iy, idx_e_valid));
        ty_vals = squeeze(Ty(ix, iy, idx_e_valid));
        
        u10_300(idx_t_seg_valid, n) = u_vals;
        v10_300(idx_t_seg_valid, n) = v_vals;
        tx_300(idx_t_seg_valid,  n) = tx_vals;
        ty_300(idx_t_seg_valid,  n) = ty_vals;
        
        % ★ Calculate delta: δτ(t) = τ(t) - τ(t-1h)
        idx_e_prev = idx_e_valid - 1;
        ok_prev = idx_e_prev >= 1;   % If not prepended and falling at the start of the month, this might fail
        
        if any(ok_prev)
            tx_prev = nan(size(tx_vals));
            ty_prev = nan(size(ty_vals));
            
            tx_prev(ok_prev) = squeeze(Tx(ix, iy, idx_e_prev(ok_prev)));
            ty_prev(ok_prev) = squeeze(Ty(ix, iy, idx_e_prev(ok_prev)));
            
            dtx_300(idx_t_seg_valid, n) = tx_vals - tx_prev;
            dty_300(idx_t_seg_valid, n) = ty_vals - ty_prev;
        end
    end
end

save('drifter_ERA5_u10v10_tx_ty_dtx_dty_300h.mat', ...
     'u10_300','v10_300','tx_300','ty_300','dtx_300','dty_300','-v7.3');
fprintf('Completed: Extracted ERA5 wind speed/stress at fixed points for all 300h segments, and calculated delta wind stress (1-hour difference).\n');