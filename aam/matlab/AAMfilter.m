%% driver for AAM algorithm
function AAMfilter(repopath, person, method, visualize_results)

datapath = sprintf(repopath, person);
settings_filename = fullfile(datapath, 'settings.txt');

[all_images, all_points] = read_settings(settings_filename);

for i=1:length(all_images)
    I{i} = imread(fullfile(datapath, all_images{i}));
    pts{i} = read_points(fullfile(datapath, all_points{i}));
    
    % scale up
    scale_factor = 1.0;
    I{i} = imresize(I{i}, scale_factor);
    pts{i} = pts{i} * scale_factor;
    
    shape{i} = reshape(pts{i}', [], 1);
end

all_shapes = cell2mat(shape);

all_aligned_shapes = all_shapes;
mean_shape = all_aligned_shapes(:,1);

for iter=1:5
    new_mean_shape = mean(all_aligned_shapes, 2);
    
    new_mean_shape = scale_shape(new_mean_shape, 100);
    
    if norm(new_mean_shape - mean_shape) < 1e-4
        break;
    end
    fprintf('%d: %.6f\n', iter, norm(new_mean_shape - mean_shape));
    
    mean_shape = new_mean_shape;
    if visualize_results
        figure; draw_shape(mean_shape);
        pause;
    end
    
    % align every shape to the mean shape
    for i=1:length(all_images)
        aligned_shape{i} = align_shape(shape{i}, mean_shape);
        if visualize_results
            figure(1); clf; hold on;
            draw_shape(mean_shape);
            draw_shape(aligned_shape{i}, 'r.');
            pause;
        end
    end
    all_aligned_shapes = cell2mat(aligned_shape);
end

% build shape model
all_aligned_shapes = cell2mat(aligned_shape);
all_aligned_shapes = all_aligned_shapes - repmat(mean_shape, 1, size(all_aligned_shapes,2));
[coeff, score, latent, tsquared, explained] = pca(all_aligned_shapes');
total_explained = cumsum(explained);
num_modes = find(total_explained>98, 1, 'first');
model.shape.num_modes = num_modes;
model.shape.x = mean_shape;
model.shape.P = coeff(:, 1:num_modes);


% create warping
tic;
mean_shape_verts = reshape(mean_shape, 2, [])';
mean_shape_tri = delaunay(mean_shape_verts(:,1), mean_shape_verts(:,2));

for i=1:length(all_images)
    xi = pts{i}(:,1);
    yi = pts{i}(:,2);
    tri_i = mean_shape_tri;
    
    if visualize_results
        figure(1); clf;
        imshow(I{i}); hold on;
        triplot(tri_i, xi, yi);
        plot(xi, yi, 'g.');
        pause;
    end
end

[h, w, d] = size(I{1});
for j=1:length(mean_shape_tri)
    tri_indices{j} = mean_shape_tri(j,:);
    tri_verts_j = mean_shape_verts(tri_indices{j},:);
    mean_mask{j} = repmat(poly2mask(tri_verts_j(:,1), tri_verts_j(:,2), h, w), [1, 1, d]);
end

mean_shape_poly_k = convhull(mean_shape_verts(:,1), mean_shape_verts(:,2));
mean_shape_poly = mean_shape_verts(mean_shape_poly_k,:);
mean_shape_region = poly2mask(mean_shape_poly(:,1), mean_shape_poly(:,2), h, w);
toc;

mask_sum = mean_mask{1};
for j=2:length(mean_mask)    
    mask_sum = mask_sum + mean_mask{j};
end
if visualize_results
    figure(3);imshow(mask_sum);
end

tic;
mean_texture = zeros(size(I{1}));

parfor i=1:length(all_images)
    t_i = tic;
    I_i = im2double(I{i});
    
    [h, w, d] = size(I_i);
    mask = zeros(size(I_i));
    masked_i = zeros(size(I_i));
    
    N = length(mean_shape_tri);
    tri_verts = cell(1,N);
    masks  = cell(1, N);
    masked = cell(1, N);    
    for j=1:length(mean_shape_tri)
        tri_ij = tri_indices{j};
        tri_verts{j} = pts{i}(tri_ij, :);
        masks{j} = poly2mask(tri_verts{j}(:,1), tri_verts{j}(:,2), h, w);
        masks{j} = repmat(masks{j}, [1, 1, d]);
        masked{j} = I_i .* masks{j};
        mask = mask + masks{j};
        masked_i = masked_i + masked{j};
    end
    
    % warped masks
    warped{i} = zeros(size(I_i));
    warped_masked = cell(1,N);
    for j=1:length(mean_shape_tri)
        r_j = imref2d(size(I_i));
        
        if false
            % warp mean_shape_verts to tri_verts{j}
        
        else
            % warp tri_verts{j} to mean_shape_verts
            tform_j = estimateGeometricTransform(tri_verts{j}, mean_shape_verts(tri_indices{j}, :), 'affine');
            warped_masked{j} = imwarp(I_i, tform_j, 'OutputView', r_j);
            warped{i} = warped{i} + warped_masked{j} .* mean_mask{j};
        end
    end
    
    if visualize_results
        figure(2); N = 3;
        subplot(1, N, 1); imshow(I{i}); axis on;
        subplot(1, N, 2); imshow(masked_i); axis on;
        subplot(1, N, 3); imshow(warped{i}); axis on;
        pause;
    end
    toc(t_i);
    mean_texture = mean_texture + warped{i};
end

mean_texture = mean_texture / length(I);
if visualize_results
    figure(3);imshow(mean_texture);pause;
end
toc;

tic;
[mean_r, mean_g, mean_b] = split_channels(mean_texture);
mean_texture_vec = [mean_r(mean_shape_region); mean_g(mean_shape_region); mean_b(mean_shape_region)];
mean_texture_vec0 = mean_texture_vec;

for iter=1:100
    new_mean_texture_vec = zeros(size(mean_texture_vec));
    N = length(mean_texture_vec) / 3;
    for i=1:length(I)
        [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
        wr = Ii_r(mean_shape_region);
        wg = Ii_g(mean_shape_region);
        wb = Ii_b(mean_shape_region);
        
        warped_vec_i = [wr; wg; wb];
        alpha_i = dot(warped_vec_i, mean_texture_vec);
        beta_i = [mean(wr) * ones(N,1); mean(wg) * ones(N, 1); mean(wb) * ones(N,1)];
        scaled_warped_vec{i} = (warped_vec_i - beta_i) / alpha_i;
        new_mean_texture_vec = new_mean_texture_vec + scaled_warped_vec{i};
    end
    new_mean_texture_vec = new_mean_texture_vec / length(I);
    fprintf('%d: %.6f, %.6f\n', iter, norm(mean_texture_vec - new_mean_texture_vec), norm(new_mean_texture_vec));
    
    if norm(mean_texture_vec - new_mean_texture_vec) < 1e-6
        break
    end
    step_alpha = 0.5;
    mean_texture_vec = mean_texture_vec * step_alpha + (1.0 - step_alpha) * new_mean_texture_vec;
    
    [final_r, final_g, final_b] = split_channels(mean_texture);
    
    final_r(mean_shape_region) = mean_texture_vec(1:N);
    final_g(mean_shape_region) = mean_texture_vec(N+1:N*2);
    final_b(mean_shape_region) = mean_texture_vec(N*2+1:end);
    final_mean_texture = cat(3, final_r, final_g, final_b);
    
    if visualize_results
        figure(1);
        subplot(1, 2, 1); imshow(mean_texture); axis equal; colorbar;
        subplot(1, 2, 2); imshow(final_mean_texture); axis equal; colorbar;
        pause;
    end
end
toc;

% build texture model
all_texture_vec = cell2mat(scaled_warped_vec);
all_texture_vec = all_texture_vec - repmat(mean_texture_vec, 1, size(all_texture_vec, 2));

%% Try different methods for detecting failure
%method = 'tournament';
%method = 'geometry';
mkdir(method);

switch method
    case 'geometry'
        fprintf('using tournament method\n');
        exclude_count = 0;
        excluded_set = [];
        current_set = 1:length(I);
        
        fit_error_thres = 10.0;
        max_fit_error_step = 100.0;
        while max_fit_error_step > fit_error_thres
            t_step=tic;
            fit_error=[];
            for j=1:length(current_set)
                i = current_set(j);
                
                % build the shape model
                samples_k = setdiff(current_set, i);
                [coeff, score, latent, tsquared, explained] = pca(all_aligned_shapes(:, samples_k)');
                total_explained = cumsum(explained);
                num_modes = find(total_explained>98, 1, 'first');
                model.shape.num_modes = num_modes;
                model.shape.x = mean_shape;
                model.shape.P = coeff(:, 1:num_modes);
                
                [normalized_shape_i, tform_i] = align_shape(shape{i}, mean_shape);
                svec = model.shape.P' * (normalized_shape_i - model.shape.x);
                
                % build only 1 models for texture
                num_models = 1;
                model.num_texture_models = num_models;
                model.texture = cell(num_models, 1);
                for k=1:num_models
                    samples_k = setdiff(current_set, i);
                    [coeff, score, latent, tsquared, explained] = pca(all_texture_vec(:, samples_k)');
                    total_explained = cumsum(explained);
                    num_modes = find(total_explained>98, 1, 'first');
                    model.texture{k}.num_modes = num_modes;
                    model.texture{k}.x = mean_texture_vec;
                    model.texture{k}.P = coeff(:, 1:num_modes);
                end
                
                [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
                Ir = Ii_r(mean_shape_region);
                Ig = Ii_g(mean_shape_region);
                Ib = Ii_b(mean_shape_region);
                
                diff_i = zeros(size(I{i}));
                syn_i = zeros(size(I{i}));
                norm_i = 0;
                max_norm_i = 0;
                for k=1:model.num_texture_models
                    % normalize the input vector first
                    normalized_texture_i = [Ir; Ig; Ib];
                    alpha_i = dot(normalized_texture_i, model.texture{k}.x);
                    beta_i = [mean(Ir) * ones(N,1); mean(Ig) * ones(N,1); mean(Ib) * ones(N,1)];
                    normalized_texture_i = (normalized_texture_i - beta_i) / alpha_i;
                    size(normalized_texture_i)
                    size(model.texture{k}.x)
                    size(model.texture{k}.P)
                    tvec = model.texture{k}.P' * (normalized_texture_i - model.texture{k}.x);
                    [s, g] = synthesize(model, svec, tvec, k);
                    
                    max_norm_i = max(max_norm_i, norm(g - normalized_texture_i));
                    norm_i = norm_i + norm(g - normalized_texture_i);
                    
                    % unnormalize the fitted vector                    
                    g = g * alpha_i + beta_i;
                    [s_r, s_g, s_b] = split_channels(mean_texture);
                    N = length(g) / 3;
                    s_r(mean_shape_region) = g(1:N);
                    s_g(mean_shape_region) = g(N+1:N*2);
                    s_b(mean_shape_region) = g(N*2+1:end);
                    norm(s_r-s_g)
                    norm(s_g-s_b)
                    syn_texture = cat(3, s_r, s_g, s_b);
                    
                    syn_i = syn_i + syn_texture;
                    diff_i = diff_i + abs(syn_texture - im2double(warped{i}));
                end
                
                syn_i = syn_i / model.num_texture_models;
                diff_i = diff_i / model.num_texture_models;
                norm_i = norm_i / model.num_texture_models;
                diff2_i = warped{i} - syn_i;
                
                syn{j} = syn_i;
                norm_shape_i = norm(normalized_shape_i - s);
                s = transformPointsInverse(tform_i, reshape(s, 2, [])');
                syn_shape{j} = s;
                
                if visualize_results
                    figure(2); M = 6;
                    subplot(1, M, 1); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');
                    subplot(1, M, 2); imshow(I{i}); hold on; plot(syn_shape{j}(:,1), syn_shape{j}(:,2), 'r.');
                    subplot(1, M, 3); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');plot(syn_shape{j}(:,1), syn_shape{j}(:,2), 'r.');
                    subplot(1, M, 4); imshow(warped{i});
                    subplot(1, M, 5); imshow(syn_i);
                    subplot(1, M, 6); imagesc(diff_i);axis equal;title(sprintf('norm = %.6f\nnorm shape = %.6f\ndiff = %.6f', norm_i, norm_shape_i, norm(diff2_i(:))));
                    pause;
                end
                                
                fit_error(j) = norm_i;
                fit_error(j) = norm_shape_i;
            end
            [max_fit_error_step, max_j] = max(fit_error);
            
            if max_fit_error_step < fit_error_thres
                break
            end
                        
            figure(2);set(gcf, 'Position', get(0,'Screensize'));
            to_exclude = current_set(max_j);
            M = 6;
            subplot(1, M, 1); imshow(I{to_exclude}); hold on; plot(pts{to_exclude}(:,1), pts{to_exclude}(:,2), 'g.');
            subplot(1, M, 2); imshow(I{to_exclude}); hold on; draw_shape(reshape(syn_shape{max_j}', [], 1), 'r.');
            subplot(1, M, 3); imshow(I{to_exclude}); hold on; plot(pts{to_exclude}(:,1), pts{to_exclude}(:,2), 'g.');draw_shape(reshape(syn_shape{max_j}', [], 1), 'r.');
            subplot(1, M, 4); imshow(warped{to_exclude});            
            subplot(1, M, 5); imshow(syn{max_j});
            diff2_i = warped{to_exclude} - syn{max_j};
            subplot(1, M, 6); imagesc(diff2_i);axis equal;title(sprintf('norm = %.6f\ndiff = %.6f', max_fit_error_step, norm(diff2_i(:))));
            
            exclude_count = exclude_count + 1;
            max_fit_error(exclude_count) = max_fit_error_step
            excluded_set(exclude_count) = to_exclude
            current_set = setdiff(current_set, [to_exclude])            
            toc(t_step);
            %pause;
            pause(1);            
        end
        
        visualize_set(I, pts, excluded_set, struct('saveit', true, 'filename', fullfile(method, [person, '_excluded'])));
        visualize_set(I, pts, current_set, struct('saveit', true, 'filename', fullfile(method, [person, '_filtered'])));        
    case 'tournament'
        fprintf('using tournament method\n');
        
        all_texture_vec = cell2mat(scaled_warped_vec);        
        
        exclude_count = 0;
        excluded_set = [];
        current_set = 1:length(I);
        
        use_single_model = false;
        
        max_fit_error_step = 100.0;
        fit_error_step_thres = 0.75;
        model_explained_factor = 80;
        while max_fit_error_step > fit_error_step_thres
            t_step=tic;
            fit_error=[];
            
            if use_single_model
                num_models = 1;
                model.num_texture_models = num_models;
                model.texture = cell(num_models, 1);
                for k=1:num_models
                    [coeff, score, latent, tsquared, explained] = pca(all_texture_vec(:, current_set)');
                    total_explained = cumsum(explained);
                    num_modes = find(total_explained>model_explained_factor, 1, 'first');
                    model.texture{k}.num_modes = num_modes;
                    model.texture{k}.x = mean_texture_vec;
                    model.texture{k}.P = coeff(:, 1:num_modes);
                end
            end
            
            parfor j=1:length(current_set)
                i = current_set(j);
                % build only 1 models
                model_j = model;
                if ~use_single_model
                    tic;
                    num_models = 1;
                    model_j.num_texture_models = num_models;
                    model_j.texture = cell(num_models, 1);
                                        
                    
                    for k=1:num_models
                        samples_k = setdiff(current_set, i);
                        [coeff, score, latent, tsquared, explained] = pca((all_texture_vec(:, samples_k) - repmat(mean_texture_vec, 1, length(samples_k)))');
                        total_explained = cumsum(explained);
                        num_modes = find(total_explained>98, 1, 'first');
                        model_j.texture{k}.num_modes = num_modes;
                        model_j.texture{k}.x = mean_texture_vec;
                        model_j.texture{k}.P = coeff(:, 1:num_modes);
                    end
                    toc;
                end
                
                [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
                Ir = Ii_r(mean_shape_region);
                Ig = Ii_g(mean_shape_region);
                Ib = Ii_b(mean_shape_region);
                
                N = length(Ir(:));
                
                diff_i = zeros(size(I{i}));
                syn_i = zeros(size(I{i}));
                norm_i = 0;
                max_norm_i = 0;
                for k=1:model_j.num_texture_models
                    % normalize the input vector first
                    normalized_texture_i = [Ir; Ig; Ib];
                    alpha_i = dot(normalized_texture_i, model_j.texture{k}.x);
                    beta_i = [mean(Ir) * ones(N,1); mean(Ig) * ones(N,1); mean(Ib) * ones(N,1)];
                    normalized_texture_i = (normalized_texture_i - beta_i) / alpha_i;
                    
                    tvec = model_j.texture{k}.P' * (normalized_texture_i - model_j.texture{k}.x);
                    [s, g] = synthesize(model_j, zeros(model_j.shape.num_modes, 1), tvec, k);
                    
                    max_norm_i = max(max_norm_i, norm(g - normalized_texture_i));
                    norm_i = norm_i + norm(g - normalized_texture_i);
                    
                    % unnormalize the fitted vector
                    g = g * alpha_i + beta_i;
                    [s_r, s_g, s_b] = split_channels(mean_texture);
                    %N = length(g) / 3;
                    s_r(mean_shape_region) = g(1:N);
                    s_g(mean_shape_region) = g(N+1:N*2);
                    s_b(mean_shape_region) = g(N*2+1:end);

                    syn_texture = cat(3, s_r, s_g, s_b);
                    
                    syn_i = syn_i + syn_texture;
                    diff_i = diff_i + abs(syn_texture - im2double(warped{i}));
                end
                
                syn_i = syn_i / model_j.num_texture_models;
                diff_i = diff_i / model_j.num_texture_models;
                norm_i = norm_i / model_j.num_texture_models;
                diff2_i = warped{i} - syn_i;
                syn{j} = syn_i;
                
                if visualize_results
                    figure(2);
                    subplot(1, 4, 1); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');
                    subplot(1, 4, 2); imshow(warped{i});
                    subplot(1, 4, 3); imshow(syn_i);
                    subplot(1, 4, 4); imagesc(diff_i);axis equal;title(sprintf('norm = %.6f\nmax norm = %.6f\ndiff = %.6f', norm_i, max_norm_i, norm(diff2_i(:))));
                    pause;
                end
                
                fit_error(j) = norm_i;
            end
            [max_fit_error_step, max_j] = max(fit_error);
            
            if max_fit_error_step < fit_error_step_thres
                break
            end
                        
            figure(2);set(gcf, 'Position', get(0,'Screensize'));
            to_exclude = current_set(max_j);
            subplot(1, 4, 1); imshow(I{to_exclude}); hold on; plot(pts{to_exclude}(:,1), pts{to_exclude}(:,2), 'g.');
            subplot(1, 4, 2); imshow(warped{to_exclude});
            subplot(1, 4, 3); imshow(syn{max_j});
            diff2_i = warped{to_exclude} - syn{max_j};
            subplot(1, 4, 4); imagesc(diff2_i);axis equal;title(sprintf('norm = %.6f\ndiff = %.6f', max_fit_error_step, norm(diff2_i(:))));
            
            exclude_count = exclude_count + 1;
            max_fit_error(exclude_count) = max_fit_error_step
            excluded_set(exclude_count) = to_exclude
            current_set = setdiff(current_set, [to_exclude])            
            toc(t_step);
            pause(1);            
        end
        
        visualize_set(I, pts, excluded_set, struct('saveit', true, 'filename', fullfile(method, [person, '_excluded'])));
        visualize_set(I, pts, current_set, struct('saveit', true, 'filename', fullfile(method, [person, '_filtered'])));
    case 'leaveoneout'
        fprintf('using leaveoneout method\n');
        for i=1:length(I)
            % build k models
            num_models = 1;
            model.num_texture_models = num_models;
            model.texture = cell(num_models, 1);
            for k=1:num_models
                samples_k = setdiff([1:size(all_texture_vec,2)], i);
                [coeff, score, latent, tsquared, explained] = pca(all_texture_vec(:, samples_k)');
                total_explained = cumsum(explained);
                num_modes = find(total_explained>75, 1, 'first');
                model.texture{k}.num_modes = num_modes;
                model.texture{k}.x = mean_texture_vec;
                model.texture{k}.P = coeff(:, 1:num_modes);
            end
            
            [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
            Ir = Ii_r(mean_shape_region);
            Ig = Ii_g(mean_shape_region);
            Ib = Ii_b(mean_shape_region);
            
            diff_i = zeros(size(I{i}));
            syn_i = zeros(size(I{i}));
            norm_i = 0;
            max_norm_i = 0;
            for k=1:model.num_texture_models
                % normalize the input vector first
                normalized_texture_i = [Ir; Ig; Ib];
                alpha_i = dot(normalized_texture_i, model.texture{k}.x);
                beta_i = [mean(Ir) * ones(N,1); mean(Ig) * ones(N,1); mean(Ib) * ones(N,1)];
                normalized_texture_i = (normalized_texture_i - beta_i) / alpha_i;
                size(normalized_texture_i)
                size(model.texture{k}.x)
                size(model.texture{k}.P)
                tvec = model.texture{k}.P' * (normalized_texture_i - model.texture{k}.x);
                [s, g] = synthesize(model, zeros(model.shape.num_modes, 1), tvec, k);
                
                max_norm_i = max(max_norm_i, norm(g - normalized_texture_i));
                norm_i = norm_i + norm(g - normalized_texture_i);
                
                % unnormalize the fitted vector
                g = g * alpha_i + beta_i;
                [s_r, s_g, s_b] = split_channels(mean_texture);
                N = length(g) / 3;
                s_r(mean_shape_region) = g(1:N);
                s_g(mean_shape_region) = g(N+1:N*2);
                s_b(mean_shape_region) = g(N*2+1:end);
                norm(s_r-s_g)
                norm(s_g-s_b)
                syn_texture = cat(3, s_r, s_g, s_b);
                
                syn_i = syn_i + syn_texture;
                diff_i = diff_i + abs(syn_texture - im2double(warped{i}));
            end
            
            syn_i = syn_i / model.num_texture_models;
            diff_i = diff_i / model.num_texture_models;
            norm_i = norm_i / model.num_texture_models;
            diff2_i = warped{i} - syn_i;
            
            figure(2);
            subplot(1, 4, 1); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');
            subplot(1, 4, 2); imshow(warped{i});
            subplot(1, 4, 3); imshow(syn_i);
            subplot(1, 4, 4); imagesc(diff_i);axis equal;title(sprintf('norm = %.6f\nmax norm = %.6f\ndiff = %.6f', norm_i, max_norm_i, norm(diff2_i(:))));
            pause;
        end
    case 'kmodels'
        fprintf('using kmodels method\n');
        % build k models
        num_models = 64;
        model.num_texture_models = num_models;
        model.texture = cell(num_models, 1);
        for k=1:num_models
            samples_k = randperm(size(all_texture_vec, 2), ceil(length(I) / 10));
            [coeff, score, latent, tsquared, explained] = pca(all_texture_vec(:, samples_k)');
            total_explained = cumsum(explained);
            num_modes = find(total_explained>98, 1, 'first');
            model.texture{k}.num_modes = num_modes;
            model.texture{k}.x = mean_texture_vec;
            model.texture{k}.P = coeff(:, 1:num_modes);
        end
        
        % build the joint model
        % for i=1:83
        %     tvec = zeros(83,1);
        %     tvec(i) = 50;
        %     [s, g] = synthesize(model, zeros(16, 1), tvec);
        %     syn_texture = mean_texture;
        %     syn_texture(mean_shape_region) = g;
        %     figure(1);imshow(syn_texture); axis equal; colorbar;
        %     pause;
        % end
        
        for i=1:length(I)
            [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
            Ir = Ii_r(mean_shape_region);
            Ig = Ii_g(mean_shape_region);
            Ib = Ii_b(mean_shape_region);
            
            diff_i = zeros(size(I{i}));
            syn_i = zeros(size(I{i}));
            norm_i = 0;
            max_norm_i = 0;
            for k=1:model.num_texture_models
                % normalize the input vector first
                normalized_texture_i = [Ir; Ig; Ib];
                alpha_i = dot(normalized_texture_i, model.texture{k}.x);
                beta_i = [mean(Ir) * ones(N,1); mean(Ig) * ones(N,1); mean(Ib) * ones(N,1)];
                normalized_texture_i = (normalized_texture_i - beta_i) / alpha_i;
                size(normalized_texture_i)
                size(model.texture{k}.x)
                size(model.texture{k}.P)
                tvec = model.texture{k}.P' * (normalized_texture_i - model.texture{k}.x);
                [s, g] = synthesize(model, zeros(16, 1), tvec, k);
                
                max_norm_i = max(max_norm_i, norm(g - normalized_texture_i));
                norm_i = norm_i + norm(g - normalized_texture_i);
                
                % unnormalize the fitted vector
                g = g * alpha_i + beta_i;
                [s_r, s_g, s_b] = split_channels(mean_texture);
                N = length(g) / 3;
                s_r(mean_shape_region) = g(1:N);
                s_g(mean_shape_region) = g(N+1:N*2);
                s_b(mean_shape_region) = g(N*2+1:end);
                norm(s_r-s_g)
                norm(s_g-s_b)
                syn_texture = cat(3, s_r, s_g, s_b);
                
                syn_i = syn_i + syn_texture;
                diff_i = diff_i + abs(syn_texture - im2double(warped{i}));
            end
            
            syn_i = syn_i / model.num_texture_models;
            diff_i = diff_i / model.num_texture_models;
            norm_i = norm_i / model.num_texture_models;
            diff2_i = warped{i} - syn_i;
            
            figure(2);
            subplot(1, 4, 1); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');
            subplot(1, 4, 2); imshow(warped{i});
            subplot(1, 4, 3); imshow(syn_i);
            subplot(1, 4, 4); imagesc(diff_i);axis equal;title(sprintf('norm = %.6f\nmax norm = %.6f\ndiff = %.6f', norm_i, max_norm_i, norm(diff2_i(:))));
            pause;
        end
    otherwise
        fprintf('unsupported method\n');
end

