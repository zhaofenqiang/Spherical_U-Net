clc;
clear;

%% caculate standatd faces and adjacency matrix
raw_sphere = mvtk_read('E:\unc\zhengwang\dataset\raw_dataset\90\neo-0004-2_lh.RigidAligned.SphereSurf.ResampledTo40K.ico5.vtk');
faces = raw_sphere.faces;
pts = raw_sphere.vertices;

pairs = [];
for i = 1:length(faces)
   pairs = [pairs;faces(i,1), faces(i,2)];
   pairs = [pairs;faces(i,1), faces(i,3)]; 
   pairs = [pairs;faces(i,2), faces(i,3)];
end

%% Caculate average distance between nodes
% max_dis = 0;
% for i = 1:length(pairs)
%     dis = norm(pts(pairs(i,1),:) - pts(pairs(i,2),:));
%     if dis > max_dis
%         max_dis = dis;
%     end
% end
% max_dis

sum_dis = 0;
for i = 1:length(pairs)
    sum_dis = sum_dis + norm(pts(pairs(i,1),:) - pts(pairs(i,2),:));
end
avg_dis = sum_dis /length(pairs);

%% Caculate all nodes rectangle neighs' (x,y,z)
coord_in_sphere = zeros(10242,25,3);
for i = 1:length(pts)
    i
    center_pt = pts(i,:);
    
    % caculate 5*5 coordinate in tangent plane
    coord_in_plane = zeros(25,3);
    if center_pt(1) ~= 0 || center_pt(2) ~= 0
        n_x = cross([0,0,1], center_pt); % normal of the new projected x-axis
        n_y = cross(center_pt, n_x); % normal of the new projected y-axis
    else
        n_x = [1,0,0];
        n_y = [0,1,0];
    end
    n_x = n_x * (avg_dis/norm(n_x));
    n_y = n_y * (avg_dis/norm(n_y));
    corner = center_pt - 2 * n_x + 2 * n_y;
    for row = 1:5
        for column = 1:5
            temp = corner - (row-1)*n_y + (column-1)*n_x;
            temp = temp * (100/norm(temp));
            coord_in_plane((row-1)*5 + column,:) = temp;
        end
    end
    
    coord_in_sphere(i,:,:) = coord_in_plane;
    %% plot grid and pts
    plot3(coord_in_sphere(:,1), coord_in_sphere(:,2), coord_in_sphere(:,3),'r.','MarkerSize',20);hold on;
    for row = 1 : 5: 25
        line([coord_in_sphere(row,1),coord_in_sphere(row+4,1)], [coord_in_sphere(row,2),coord_in_sphere(row+4,2)], [coord_in_sphere(row,3),coord_in_sphere(row+4,3)], 'Color','red','LineStyle','-'); hold on;
    end
    for co = 1 : 5
        line([coord_in_sphere(co,1),coord_in_sphere(co+20,1)], [coord_in_sphere(co,2),coord_in_sphere(co+20,2)], [coord_in_sphere(co,3),coord_in_sphere(co+20,3)], 'Color','red','LineStyle','-'); hold on;
    end
    [x,y,z] = sphere(50);
    mesh(100*x,100*y,100*z);
    axis equal;hold on;

end
    
%% Caculate index for neighs
indices_10242 = zeros(10242,25,3);
for i = 1:length(pts)
    i
    neighs = squeeze(coord_in_sphere(i,:,:));
   
    for j = 1:length(neighs)
        neigh = neighs(j,:);
        neigh = repmat(neigh, [10242,1]);
        dis = pts - neigh;
        dis = sum(dis.*dis, 2);
        [~,index_order] = sort(dis);
        index = index_order(1:3)';
        indices_10242(i,j,:) = index;
    end  
    
end
save(strcat('E:\unc\zhengwang\dataset\format_dataset', '\rec_neigh_indices_10242.mat'), 'indices_10242'); 


%% 2562
avg_dis_2562 = 2 * avg_dis;
coord_in_sphere = zeros(2562,25,3);
for i = 1:2562
    center_pt = pts(i,:);
    
    % caculate 5*5 coordinate in tangent plane
    coord_in_plane = zeros(25,3);
    if center_pt(1) ~= 0 || center_pt(2) ~= 0
        n_x = cross([0,0,1], center_pt); % normal of the new projected x-axis
        n_y = cross(center_pt, n_x); % normal of the new projected y-axis
    else
        n_x = [1,0,0];
        n_y = [0,1,0];
    end
    n_x = n_x * (avg_dis_2562/norm(n_x));
    n_y = n_y * (avg_dis_2562/norm(n_y));
    corner = center_pt - 2 * n_x + 2 * n_y;
    for row = 1:5
        for column = 1:5
            temp = corner - (row-1)*n_y + (column-1)*n_x;
            temp = temp * (100/norm(temp));
            coord_in_plane((row-1)*5 + column,:) = temp;
        end
    end
    
    coord_in_sphere(i,:,:) = coord_in_plane;
end

indices_2562 = zeros(2562,25,3);
delete_indices = [2563:10242];
for i = 1:2562
    neighs = squeeze(coord_in_sphere(i,:,:));
    for j = 1:length(neighs)
        neigh = neighs(j,:);
        neigh = repmat(neigh, [2562,1]);
        dis = pts(1:2562,:) - neigh;
        dis = sum(dis.*dis, 2);
        [~,index_order] = sort(dis);
        index = index_order(1:3)';
        indices_2562(i,j,:) = index;
    end  
end
save(strcat('E:\unc\zhengwang\dataset\format_dataset', '\rec_neigh_indices_2562.mat'), 'indices_2562'); 



%% 642
avg_dis_642 = 2 * avg_dis_2562;
coord_in_sphere = zeros(642,25,3);
for i = 1:642
    center_pt = pts(i,:);
    
    % caculate 5*5 coordinate in tangent plane
    coord_in_plane = zeros(25,3);
    if center_pt(1) ~= 0 || center_pt(2) ~= 0
        n_x = cross([0,0,1], center_pt); % normal of the new projected x-axis
        n_y = cross(center_pt, n_x); % normal of the new projected y-axis
    else
        n_x = [1,0,0];
        n_y = [0,1,0];
    end
    n_x = n_x * (avg_dis_642/norm(n_x));
    n_y = n_y * (avg_dis_642/norm(n_y));
    corner = center_pt - 2 * n_x + 2 * n_y;
    for row = 1:5
        for column = 1:5
            temp = corner - (row-1)*n_y + (column-1)*n_x;
            temp = temp * (100/norm(temp));
            coord_in_plane((row-1)*5 + column,:) = temp;
        end
    end
    coord_in_sphere(i,:,:) = coord_in_plane;
end

indices_642 = zeros(642,25,3);
for i = 1:642
    neighs = squeeze(coord_in_sphere(i,:,:));
   
    for j = 1:length(neighs)
        neigh = neighs(j,:);
        neigh = repmat(neigh, [642,1]);
        dis = pts(1:642,:) - neigh;
        dis = sum(dis.*dis, 2);
        [~,index_order] = sort(dis);
        index = index_order(1:3)';
        indices_642(i,j,:) = index;
    end  
    
end
save(strcat('E:\unc\zhengwang\dataset\format_dataset', '\rec_neigh_indices_642.mat'), 'indices_642'); 



%% 162
avg_dis_162 = 2 * avg_dis_642;
coord_in_sphere = zeros(162,25,3);
for i = 1:162
    center_pt = pts(i,:);
    
    % caculate 5*5 coordinate in tangent plane
    coord_in_plane = zeros(25,3);
    if center_pt(1) ~= 0 || center_pt(2) ~= 0
        n_x = cross([0,0,1], center_pt); % normal of the new projected x-axis
        n_y = cross(center_pt, n_x); % normal of the new projected y-axis
    else
        n_x = [1,0,0];
        n_y = [0,1,0];
    end
    n_x = n_x * (avg_dis_162/norm(n_x));
    n_y = n_y * (avg_dis_162/norm(n_y));
    corner = center_pt - 2 * n_x + 2 * n_y;
    for row = 1:5
        for column = 1:5
            temp = corner - (row-1)*n_y + (column-1)*n_x;
            temp = temp * (100/norm(temp));
            coord_in_plane((row-1)*5 + column,:) = temp;
        end
    end
    coord_in_sphere(i,:,:) = coord_in_plane;
end

indices_162 = zeros(162,25,3);
for i = 1:162
    neighs = squeeze(coord_in_sphere(i,:,:));
   
    for j = 1:length(neighs)
        neigh = neighs(j,:);
        neigh = repmat(neigh, [162,1]);
        dis = pts(1:162,:) - neigh;
        dis = sum(dis.*dis, 2);
        [~,index_order] = sort(dis);
        index = index_order(1:3)';
        indices_162(i,j,:) = index;
    end  
    
end
save(strcat('E:\unc\zhengwang\dataset\format_dataset', '\rec_neigh_indices_162.mat'), 'indices_162'); 


%% 42
avg_dis_42 = 2 * avg_dis_162;
coord_in_sphere = zeros(42,25,3);
for i = 1:42
    center_pt = pts(i,:);
    
    % caculate 5*5 coordinate in tangent plane
    coord_in_plane = zeros(25,3);
    if center_pt(1) ~= 0 || center_pt(2) ~= 0
        n_x = cross([0,0,1], center_pt); % normal of the new projected x-axis
        n_y = cross(center_pt, n_x); % normal of the new projected y-axis
    else
        n_x = [1,0,0];
        n_y = [0,1,0];
    end
    n_x = n_x * (avg_dis_42/norm(n_x));
    n_y = n_y * (avg_dis_42/norm(n_y));
    corner = center_pt - 2 * n_x + 2 * n_y;
    for row = 1:5
        for column = 1:5
            temp = corner - (row-1)*n_y + (column-1)*n_x;
            temp = temp * (100/norm(temp));
            coord_in_plane((row-1)*5 + column,:) = temp;
        end
    end
    coord_in_sphere(i,:,:) = coord_in_plane;
end

indices_42 = zeros(42,25,3);
for i = 1:42
    neighs = squeeze(coord_in_sphere(i,:,:));
   
    for j = 1:length(neighs)
        neigh = neighs(j,:);
        neigh = repmat(neigh, [42,1]);
        dis = pts(1:42,:) - neigh;
        dis = sum(dis.*dis, 2);
        [~,index_order] = sort(dis);
        index = index_order(1:3)';
        indices_42(i,j,:) = index;
    end  
    
end
save(strcat('E:\unc\zhengwang\dataset\format_dataset', '\rec_neigh_indices_42.mat'), 'indices_42'); 