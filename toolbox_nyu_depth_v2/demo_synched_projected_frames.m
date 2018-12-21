% The directory where you extracted the raw dataset.
datasetDir = '/media/tung/General/bathrooms_part1';

files = dir(datasetDir);
for i = 11 : numel(files)
    sceneName = files(i).name;
% The name of the scene to demo.
% sceneName = 'bathroom_0009';

% The absolute directory of the 
sceneDir = sprintf('%s/%s', datasetDir, sceneName);


% Reads the list of frames.
frameList = get_synched_frames(sceneDir);

% Displays each pair of synchronized RGB and Depth frames.

%fid = fopen("text_file.txt", "w");

    
for ii = 1  : numel(frameList)
  imgRgb = imread([sceneDir '/' frameList(ii).rawRgbFilename]);
  imgDepthRaw = swapbytes(imread([sceneDir '/' frameList(ii).rawDepthFilename]));
  
  imgDepthUint= uint16(imgDepthRaw)
  dimg = fill_depth_colorization(imgRgb, imgDepthUint)
  
  
  figure(1);
  % Show the RGB image.
  subplot(1,3,1);
  imagesc(imgRgb);
  axis off;
  axis equal;
  title('RGB');
  
  % Show the Raw Depth image.
  subplot(1,3,2);
  imagesc(dimg);
  axis off;
  axis equal;
  title('Raw Depth');
  caxis([800 1100]);
  
  % Show the projected depth image.
  imgDepthProj = project_depth_map(imgDepthRaw, imgRgb);
  subplot(1,3,3);
  imagesc(imgDepthProj);
  axis off;
  axis equal;
  title('Projected Depth');
  
  pause(0.01);
 
  dimg_filename = [sceneDir '/denoised-' frameList(ii).rawDepthFilename] %
  dimg_filename_1 = strrep(dimg_filename, '.pgm', '.png')
  imwrite(dimg,dimg_filename_1)
  
  imgRgb_filename = [sceneDir '/denoised-' frameList(ii).rawRgbFilename]
  imgRgb_filename_1 = strrep(imgRgb_filename, '.ppm', '.png')
  imwrite(imgRgb, imgRgb_filename_1)
  
  %string_to_write = strcat(imgRgb_filename_1,'	')
  %string_to_write_1 = strcat(string_to_write, dimg_filename_1)
  %string_to_write_2 = (string_to_write_1 + "\n")
  
  %fprintf(fid, string_to_write_2);
  
end
end


%fclose(fid);
    
    
