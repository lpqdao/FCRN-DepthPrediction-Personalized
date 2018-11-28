  GNU nano 2.2.6                                                                                                                                                                                                                                                                           File: error_metrics.m

function results = error_metrics(pred, gt, mask,imdb, net, varargin)

% Compute error metrics on benchmark datasets
% -------------------------------------------------------------------------

% make sure predictions and ground truth have same dimensions
if size(pred) ~= size(gt)
    pred = imresize(pred, [size(gt,1), size(gt,2)], 'bilinear');
end

if isempty(mask)
    n_pxls = numel(gt);
else
    n_pxls = sum(mask(:));  % average over valid pixels only
end

fprintf('\n Errors computed over the entire test set \n');
fprintf('------------------------------------------\n');

opts.gpu = false;           % Set to true (false) for GPU (CPU only) support
opts.plot = true;          % Set to true to visualize the predictions during inference
opts = vl_argparse(opts, varargin);

% Set network properties
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
out = net.getVarIndex('prediction');
if opts.gpu
    net.move('gpu');
end

images = imdb.images;
images = imresize(images, net.meta.normalization.imageSize(1:2)); % for multiple images loop over the image files.
groundTruth = [];
montage(images(:,:,:,randperm(134, 6)))

varSizes = net.getVarSizes({'data', net.meta.normalization.imageSize});
relArr = zeros(varSizes{out}(1), varSizes{out}(2), varSizes{out}(3), size(images, 4));    % initiliaze
rmsArr = zeros(varSizes{out}(1), varSizes{out}(2), varSizes{out}(3), size(images, 4));    % initiliaze
lg10Arr= zeros(varSizes{out}(1), varSizes{out}(2), varSizes{out}(3), size(images, 4));    % initiliaze
% Mean Absolute Relative Error
relArr = abs(gt(:) - pred(:)) ./ gt(:);    % compute errors
relArr(~mask) = 0;                         % mask out invalid ground truth pixels
rel = sum(relArr) / n_pxls;                % average over all pixels

results.relArr = relArr;


%rel(~mask) = 0;                         % mask out invalid ground truth pixels
%rel = sum(rel) / n_pxls;                % average over all pixels
%fprintf('Mean Absolute Relative Error: %4f\n', rel);

% Root Mean Squared Error
rmsArr = (gt(:) - pred(:)).^2;
rmsArr(~mask) = 0;
%rms = sqrt(sum(rms) / n_pxls);

%rms = (gt(:) - pred(:)).^2;
%rms(~mask) = 0;
rms = sqrt(sum(rmsArr) / n_pxls);
%fprintf('Root Mean Squared Error: %4f\n', rms);

% LOG10 Error
lg10Arr = abs(log10(gt(:)) - log10(pred(:)));
lg10Arr(~mask) = 0;
lg10 = sum(lg10Arr) / n_pxls ;

%lg10 = abs(log10(gt(:)) - log10(pred(:)));
%lg10(~mask) = 0;
%lg10 = sum(lg10) / n_pxls ;
%fprintf('Mean Log10 Error: %4f\n', lg10);

%results.rel = rel;
results.rmsArr = rmsArr;
results.log10Arr = lg10Arr;

%relArr(:,:,i)= gather(net.vars(out).value);
%rmsArr(:,:,i) =  gather(net.vars(out).value);
%lg10Arr(:,:,i) = gather(net.vars(out).value);
i=n_pxls;

%visualize results
if opts.plot
    colormap jet
    if ~isempty(relArr)
        subplot(1,3,1), imagesc(relArr(:,:,i)), title('Mean Absolute Relative Error'), axis off
        subplot(1,3,2), imagesc(rmsArr(:,:,i)), title('RMS Error'), axis off
        subplot(1,3,3), imagesc(lg10Arr(:,:,i)), title('Log10 Error'), axis off
    else
        subplot(1,2,1), imagesc(rmsArr(:,:,i)), title('RMS Error'), axis off
        subplot(1,2,2), imagesc(lg10Arr(:,:,i)), title('Log10 Error'), axis off
    end
    drawnow;
end

rel=abs(gt-pred)./gt;
rms=(gt-pred).^2;
lg10=abs(log(gt)-log(pred));
colormap jet
imagesc(uint8(images(:,:,:,i)));

tensor_546=table2array(frame546);
gt_546=gt(:,:,546);
rel_546= abs(gt_546-tensor_546)./gt_546;
imagesc(rel_546(:,:));

imagesc(gt(:,:,i));
imagesc(pred(:,:,i));
imagesc(rel(:,:,i));
imagesc(rms(:,:,i));
imagesc(lg10(:,:,i));


