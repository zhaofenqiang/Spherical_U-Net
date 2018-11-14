clc;
clear;

%% caculate standatd faces and adjacency matrix
sphere_163842 = mvtk_read('/media/fenqiang/DATA/unc/Data/ForFenqiang/Year0/neo-0004-2-T/lh.AlignedToUNCAtlasResampled.SphereSurf.vtk');
faces = sphere_163842.faces;
pts = sphere_163842.vertices;

pairs = zeros(983040,2);
for i = 1:length(faces)
   pairs((i-1)*3+1, :) = [faces(i,1), faces(i,2)];
   pairs((i-1)*3+2, :) = [faces(i,1), faces(i,3)]; 
   pairs((i-1)*3+3, :) = [faces(i,2), faces(i,3)];
end

%% 163842 nodes' adj_mat
adj_mat_163842 = zeros(length(pts),6);
for i = 1:length(pairs)
    row_index = pairs(i,1);
    if ~ismember(pairs(i,2), adj_mat_163842(row_index,:))
        if ismember(0, adj_mat_163842(row_index,:))
            column_index = find(adj_mat_163842(row_index,:)==0);
            adj_mat_163842(row_index, column_index(1)) = pairs(i,2);
        else
            i
        end
    end
    row_index = pairs(i,2);
    if ~ismember(pairs(i,1), adj_mat_163842(row_index,:))
        if ismember(0, adj_mat_163842(row_index,:))
            column_index = find(adj_mat_163842(row_index,:)==0);
            adj_mat_163842(row_index, column_index(1)) = pairs(i,1);
        else
            i
        end
    end
end

%% 40962 adj_mat and its orders
adj_mat_40962 = zeros(40962,6);
for i = 1:40962
    for j = 1:6
        delete_neigh = adj_mat_163842(i,j);
        if delete_neigh == 0
            new_neigh = i;
        else
            neigh_of_delete_neigh = adj_mat_163842(delete_neigh, :);
            for k = 1:6
                if neigh_of_delete_neigh(k) < 40963 && neigh_of_delete_neigh(k) ~= i
                    new_neigh = neigh_of_delete_neigh(k);
                end
            end              
        end
        adj_mat_40962(i,j) = new_neigh;
    end   
end

%% 40962 faces
faces_40962 = [];
for i = 1:40962
    i1 = i;
    if i < 13
        temp = 5;
    else
        temp = 6;
    end
    for j = 1:temp
        if adj_mat_40962(i,j) > i1
            i2 = adj_mat_40962(i,j);
            if i2 < 13
                temp2 = 5;
            else
                temp2 = 6;
            end
            for k = 1:temp2
                if adj_mat_40962(adj_mat_40962(i,j),k) > i1 &&   adj_mat_40962(adj_mat_40962(i,j),k) > i2
                    if ismember(i1, adj_mat_40962( adj_mat_40962(adj_mat_40962(i,j),k),:))
                       i3 = adj_mat_40962(adj_mat_40962(i,j),k);
                       faces_40962 = [faces_40962;i1,i2,i3];
                    end
                end
            end
        end
    end
end


%% write predicted attrabutes to sphere
raw_surface = mvtk_read('/media/fenqiang/DATA/unc/Data/SmoothedThickness/Year1/T0085-1-2-1yr_lh.AlignedToUNCAtlasResampled.SphereSurf.vtk');

predict =  strcat('/home/fenqiang/Spherical_U-Net/pred/pred_0.txt');
pred = load(predict);

pred = struct('vertices',raw_surface.vertices(1:40962,:), 'faces',faces_40962, 'sulc', raw_surface.sulc(1:40962),'thickness',raw_surface.thickness(1:40962), 'pred_sulc', pred);
mvtk_write(pred, strcat('/media/fenqiang/DATA/unc/Data/predicted/', 'T0085-1-2.vtk'), 'legacy-ascii')

