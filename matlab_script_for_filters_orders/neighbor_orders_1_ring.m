clc;
clear;

%% caculate standatd faces and adjacency matrix
sphere_163842 = mvtk_read('/media/fenqiang/DATA/unc/Data/Template/sphere_163842.vtk');
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

adj_mat_163842_for_nitrc = adj_mat_163842;
for i = 1:12
    adj_mat_163842_for_nitrc(i,6) = i;
end
adj_mat_order = zeros(length(adj_mat_163842_for_nitrc), 6);
for i = 1:length(adj_mat_163842_for_nitrc)
    neighs = pts(adj_mat_163842_for_nitrc(i,:), :); 
    center_pt = pts(i,:);
    neighs_angle = compute_angles(neighs, center_pt);
    neighs_angle = neighs_angle + pi/4;
    neighs_angle = mod(neighs_angle, 2*pi);
    [~, temp] = sort(neighs_angle);
    adj_mat_order(i,:) = adj_mat_163842_for_nitrc(i,temp);
end
save(strcat('/home/fenqiang/Spherical_U-Net/neigh_indices', '/adj_mat_order_163842.mat'), 'adj_mat_order'); 

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

adj_mat_order = zeros(length(adj_mat_40962), 6);
for i = 1:length(adj_mat_40962)
    neighs = pts(adj_mat_40962(i,:), :); 
    center_pt = pts(i,:);
    neighs_angle = compute_angles(neighs, center_pt);
    neighs_angle = neighs_angle + pi/4;
    neighs_angle = mod(neighs_angle, 2*pi);
    [~, temp] = sort(neighs_angle);
    adj_mat_order(i,:) = adj_mat_40962(i,temp);
end

save(strcat('/home/fenqiang/Spherical_U-Net/neigh_indices', '/adj_mat_order_40962.mat'), 'adj_mat_order'); 


%% others 
adj_mat_intermediate = adj_mat_40962;
nums = [10242, 2562, 642, 162, 42, 12];
for n = 1:length(nums)
   
    num = nums(n);
    adj_mat = zeros(num,6);
    for i = 1:num
        for j = 1:6
            delete_neigh = adj_mat_intermediate(i,j);
            if delete_neigh == i
                new_neigh = i;
            else
                neigh_of_delete_neigh = adj_mat_intermediate(delete_neigh, :);
                for k = 1:6
                    if neigh_of_delete_neigh(k) < num+1 && neigh_of_delete_neigh(k) ~= i
                        new_neigh = neigh_of_delete_neigh(k);
                    end
                end              
            end
            adj_mat(i,j) = new_neigh;
        end   
    end

    adj_mat_order = zeros(length(adj_mat), 6);
    for i = 1:length(adj_mat)
        neighs = pts(adj_mat(i,:), :); 
        center_pt = pts(i,:);
        neighs_angle = compute_angles(neighs, center_pt);
        neighs_angle = neighs_angle + pi/4;
        neighs_angle = mod(neighs_angle, 2*pi);
        [~, temp] = sort(neighs_angle);
        adj_mat_order(i,:) = adj_mat(i,temp);
    end
    
    adj_mat_intermediate = adj_mat;
    save(strcat('/home/fenqiang/Spherical_U-Net/neigh_indices', '/adj_mat_order_', num2str(num), '.mat'), 'adj_mat_order'); 
    
end