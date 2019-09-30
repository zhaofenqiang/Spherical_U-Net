clc;
clear;

%% extract standard parcellation labels
stdsphere = mvtk_read('E:\unc\zhengwang\dataset\raw_dataset\90\neo-0004-2_lh.RigidAligned.SphereSurf.ResampledTo40K.ico5.vtk');
stdParVec =  unique(stdsphere.par_vec_fs, 'rows');

%% extract 3 features and save
path = 'E:\unc\zhengwang\dataset\raw_dataset\90';
subjectList = dir(path);
for i = 3:length(subjectList)
    subjectName = subjectList(i).name;
    subjectPath = strcat(path, '\', subjectName);
  
    data = mvtk_read(subjectPath);
    curvatureData = data.curv;
    EucDepthData = data.depth;
    sulcData = data.sulc;
    parData = data.par_vec_fs;
    label = [];
    for k = 1:length(parData)
        [p,q] = ismember(parData(k,:), stdParVec, 'rows');
        if p
           label = [label,q];
        else
            error('cannot find label.');
        end
    end
    data = [curvatureData, EucDepthData, sulcData];
    save(strcat('E:\unc\zhengwang\dataset\format_dataset\90', '\', subjectName, '.mat'), 'data'); 
    save(strcat('E:\unc\zhengwang\dataset\format_dataset\90', '\', subjectName, '.label'), 'label'); 
end