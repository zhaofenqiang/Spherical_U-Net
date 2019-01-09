clc;
clear;

DataPath_Y0 = '/media/fenqiang/DATA/unc/Data/ForFenqiang/Year0';
DataPath_Y1 = '/media/fenqiang/DATA/unc/Data/ForFenqiang/Year1';
DataPath_Y2 = '/media/fenqiang/DATA/unc/Data/ForFenqiang/Year2';

%% search all subjects' names both in year0 and year1
nameList_Y1{1} = '';
allnames = dir(DataPath_Y1);
for i = 3:length(allnames)
    subjectName = allnames(i).name;
    nameList_Y1 = {nameList_Y1{1,:}, subjectName};  
end
nameList_Y1 = nameList_Y1(2:end)';
for i = 1:length(nameList_Y1)
    if isempty(strfind(nameList_Y1{i}, 'T0'))
        nameList_Y1{i} = nameList_Y1{i}(1:10);
    else
        nameList_Y1{i} = nameList_Y1{i}(1:9);
    end
end

% count = [];
% for i = 1:length(nameList_Y1)
%    a = nameList_Y1{i};
%    local_count = 0;
%    for j = 1:length(nameList_Y1)
%       if strfind(nameList_Y1{j}, a)
%           local_count = local_count +1;
%       end
%    end
%    count = [count; local_count];    
% end

nameList_Y0{1} = '';
allnames = dir(DataPath_Y0);
for i = 3:length(allnames)
    subjectName = allnames(i).name;
    nameList_Y0 = {nameList_Y0{1,:}, subjectName};
end
nameList_Y0 = nameList_Y0(2:end)';
for i = 1:length(nameList_Y0)
    if isempty(strfind(nameList_Y0{i}, 'T0'))
        nameList_Y0{i} = nameList_Y0{i}(1:10);
    else
        nameList_Y0{i} = nameList_Y0{i}(1:9);
    end
end

candiNames_Y1{1} = ''; 
for i = 1:length(nameList_Y1)
    if ismember(nameList_Y1{i}, nameList_Y0)
        candiNames_Y1 = {candiNames_Y1{1,:}, nameList_Y1{i}};
    end
end
candiNames_Y1 = candiNames_Y1(2:end)';
% 
% count = [];
% for i = 1: length(candiNames_Y1)
%     a = 0;
%     b = candiNames_Y1{i};
%     for j =1: length(nameList_Y1)
%         if strfind(b, nameList_Y1{j})    
%             a = a + 1;
%         end
%     end
%     count = [count; a];
% end

%% extract features
allNamesInY0 = dir(DataPath_Y0);
allNamesInY0 = {allNamesInY0.name}';
allNamesInY1 = dir(DataPath_Y1);
allNamesInY1 = {allNamesInY1.name}';
for i = 1:length(candiNames_Y1)
    subjectName = candiNames_Y1{i};
    if ~ismember(subjectName, allNamesInY0)
        subjectName = strcat(candiNames_Y1{i}, '-T');
        if ~ismember(subjectName, allNamesInY0)
            subjectName = strcat(candiNames_Y1{i}, '-1');
            if ~ismember(subjectName, allNamesInY0)
                subjectName = strcat(candiNames_Y1{i}, '-2');
                if ~ismember(subjectName, allNamesInY0)
                    subjectName
                end
            end
        end
    end

    Y0Path = strcat('/media/fenqiang/DATA/unc/Data/SmoothedThickness/Year0/', subjectName);
    for left2right = 1:2
        if left2right == 1
            sphere = 'lh.AlignedToUNCAtlasResampled.SphereSurf.vtk';
        else
            sphere = 'rh.AlignedToUNCAtlasResampled.SphereSurf.vtk';
        end
        data = mvtk_read(strcat(Y0Path, '_', sphere));
        curvData = data.curv(1:40962);
        sulcData = data.sulc(1:40962);
        thicknessData = data.thickness(1:40962);
        data = [curvData, sulcData, thicknessData];
        save(strcat('/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed', '/', candiNames_Y1{i}, '_', sphere, '.Y0'), 'data');
    end
    
    subjectName = strcat(candiNames_Y1{i}, '-1yr');
    if ~ismember(subjectName, allNamesInY1)
        subjectName = strcat(candiNames_Y1{i}, '-1-1year');
        if ~ismember(subjectName, allNamesInY1)
            subjectName = strcat(candiNames_Y1{i}, '-1year');   
            if ~ismember(subjectName, allNamesInY1)
                subjectName
            end
        end
    end
    Y1Path = strcat('/media/fenqiang/DATA/unc/Data/SmoothedThickness/Year1/', subjectName);
    for left2right = 1:2
        if left2right == 1
            sphere = 'lh.AlignedToUNCAtlasResampled.SphereSurf.vtk';
        else
            sphere = 'rh.AlignedToUNCAtlasResampled.SphereSurf.vtk';
        end
        data = mvtk_read(strcat(Y1Path, '_', sphere));
        curvData = data.curv(1:40962);
        sulcData = data.sulc(1:40962);
        thicknessData = data.thickness(1:40962);
        data = [curvData, sulcData, thicknessData];
        save(strcat('/media/fenqiang/DATA/unc/Data/MissingDataPredictionSmoothed', '/', candiNames_Y1{i}, '_', sphere, '.Y1'), 'data');
    end
    
end