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

adj_mat_order_1ring = zeros(length(adj_mat_40962), 6);
for i = 1:length(adj_mat_40962)
    neighs = pts(adj_mat_40962(i,:), :); 
    center_pt = pts(i,:);
    neighs_angle = compute_angles(neighs, center_pt);
    neighs_angle = neighs_angle + pi/8;
    neighs_angle = mod(neighs_angle, 2*pi);
    [~, temp] = sort(neighs_angle);
    adj_mat_order_1ring(i,:) = adj_mat_40962(i,temp);
end


adj_mat_2ring = zeros(40962,12);
for i = 1:40962
    if i < 13
        one_ring_neigh = adj_mat_40962(i,1:5);
    else
        one_ring_neigh = adj_mat_40962(i,:);
    end
    
    two_ring_neigh = [];
    for j = 1:size(one_ring_neigh,2)
        a = setdiff(adj_mat_40962(one_ring_neigh(j),:), [adj_mat_40962(i,:),i]);
        two_ring_neigh = [two_ring_neigh, a];
    end
    two_ring_neigh = unique(two_ring_neigh);
    
    if size(two_ring_neigh, 2) == 10
        two_ring_neigh = [two_ring_neigh,i,i];
    elseif size(two_ring_neigh, 2) == 11
        two_ring_neigh = [two_ring_neigh,i];
    else
        assert(size(two_ring_neigh, 2) == 12, 'not 12 2ring neighbors')
    end
    
    adj_mat_2ring(i,:) = two_ring_neigh;
end

adj_mat_order_2ring = zeros(size(adj_mat_2ring, 1), 12);
for i = 1:size(adj_mat_2ring, 1)
    neighs = pts(adj_mat_2ring(i,:), :); 
    center_pt = pts(i,:);
    neighs_angle = compute_angles(neighs, center_pt);
    neighs_angle = neighs_angle + pi/8;
    neighs_angle = mod(neighs_angle, 2*pi);
    [~, temp] = sort(neighs_angle);
    adj_mat_order_2ring(i,:) = adj_mat_2ring(i,temp);
end
adj_mat_order_2ring = [adj_mat_order_1ring, adj_mat_order_2ring];

save(strcat('/home/fenqiang/Spherical_U-Net/neigh_indices', '/adj_mat_order_2ring_40962.mat'), 'adj_mat_order_2ring');


%% others 
adj_mat_1ring_intermediate = adj_mat_40962;
nums = [10242, 2562, 642, 162, 42, 12];
for n = 1:length(nums)
   
    num = nums(n);
    adj_mat_1ring = zeros(num,6);
    for i = 1:num
        for j = 1:6
            delete_neigh = adj_mat_1ring_intermediate(i,j);
            if delete_neigh == i
                new_neigh = i;
            else
                neigh_of_delete_neigh = adj_mat_1ring_intermediate(delete_neigh, :);
                for k = 1:6
                    if neigh_of_delete_neigh(k) < num+1 && neigh_of_delete_neigh(k) ~= i
                        new_neigh = neigh_of_delete_neigh(k);
                    end
                end              
            end
            adj_mat_1ring(i,j) = new_neigh;
        end   
    end

    adj_mat_order_1ring = zeros(length(adj_mat_1ring), 6);
    for i = 1:length(adj_mat_1ring)
        neighs = pts(adj_mat_1ring(i,:), :); 
        center_pt = pts(i,:);
        neighs_angle = compute_angles(neighs, center_pt);
        neighs_angle = neighs_angle + pi/8;
        neighs_angle = mod(neighs_angle, 2*pi);
        [~, temp] = sort(neighs_angle);
        adj_mat_order_1ring(i,:) = adj_mat_1ring(i,temp);
    end
    
    adj_mat_2ring = zeros(num,12);
    for i = 1:num
        if i < 13
            one_ring_neigh = adj_mat_1ring(i,1:5);
        else
            one_ring_neigh = adj_mat_1ring(i,:);
        end

        two_ring_neigh = [];
        for j = 1:size(one_ring_neigh,2)
            a = setdiff(adj_mat_1ring(one_ring_neigh(j),:), [adj_mat_1ring(i,:),i]);
            two_ring_neigh = [two_ring_neigh, a];
        end
        two_ring_neigh = unique(two_ring_neigh);

        if size(two_ring_neigh, 2) == 10
            two_ring_neigh = [two_ring_neigh,i,i];
        elseif size(two_ring_neigh, 2) == 11
            two_ring_neigh = [two_ring_neigh,i];
        else
            assert(size(two_ring_neigh, 2) == 12, 'not 12 2ring neighbors')
        end

        adj_mat_2ring(i,:) = two_ring_neigh;
    end

    adj_mat_order_2ring = zeros(size(adj_mat_2ring, 1), 12);
    for i = 1:size(adj_mat_2ring, 1)
        neighs = pts(adj_mat_2ring(i,:), :); 
        center_pt = pts(i,:);
        neighs_angle = compute_angles(neighs, center_pt);
        neighs_angle = neighs_angle + pi/8;
        neighs_angle = mod(neighs_angle, 2*pi);
        [~, temp] = sort(neighs_angle);
        adj_mat_order_2ring(i,:) = adj_mat_2ring(i,temp);
    end
    adj_mat_order_2ring = [adj_mat_order_1ring, adj_mat_order_2ring];

    adj_mat_1ring_intermediate = adj_mat_1ring;
    
    save(strcat('/home/fenqiang/Spherical_U-Net/neigh_indices', '/adj_mat_order_2ring_', num2str(num), '.mat'), 'adj_mat_order_2ring'); 
    
end