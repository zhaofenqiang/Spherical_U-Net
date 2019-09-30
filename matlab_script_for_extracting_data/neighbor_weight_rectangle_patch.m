clc;
clear;

%% caculate standatd faces and adjacency matrix and average distance between nodes
raw_sphere = mvtk_read('E:\unc\zhengwang\dataset\raw_dataset\90\neo-0004-2_lh.RigidAligned.SphereSurf.ResampledTo40K.ico5.vtk');
faces = raw_sphere.faces;
pts = raw_sphere.vertices;

pairs = [];
for i = 1:length(faces)
   pairs = [pairs;faces(i,1), faces(i,2)];
   pairs = [pairs;faces(i,1), faces(i,3)]; 
   pairs = [pairs;faces(i,2), faces(i,3)];
end

sum_dis = 0;
for i = 1:length(pairs)
    sum_dis = sum_dis + norm(pts(pairs(i,1),:) - pts(pairs(i,2),:));
end
avg_dis = sum_dis /length(pairs)/2;


%% caculate weight
loop = [10242,2562,642,162,42];
for l = 1:5
    vertices = loop(l);
    load(strcat('E:\unc\zhengwang\dataset\format_dataset\rec_neigh_indices_',num2str(vertices)));
    
    if l == 1 
        indices = indices_10242;
    elseif l ==2
        indices = indices_2562;
    elseif l ==3
        indices = indices_642;
    elseif l ==4
        indices = indices_162;
    elseif l ==5
        indices = indices_42;
    end
    
    weight = zeros(size(indices));
    avg_dis = avg_dis * 2;
    for i = 1:length(indices)
        i
        center_pt = pts(i,:);
        neighs = squeeze(indices(i,:,:));

        coord_in_plane = zeros(25,3); % caculate the projected point's coordinate on the sphere
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

        dis = zeros(1,3);
        for j = 1:length(neighs)
            neigh = neighs(j,:);
            dis = [pts(neigh(1),:);pts(neigh(2),:);pts(neigh(3),:)] - repmat(coord_in_plane(j,:), [3,1]);
            dis = sqrt(sum(dis.*dis, 2));
            if all(dis)
                dis = 1./dis;
                weight(i,j,:) = [dis(1)/sum(dis),dis(2)/sum(dis),dis(3)/sum(dis)];
            else
                temp = find(dis == 0);
                dis = zeros(1,3);
                dis(temp) = 1
                weight(i,j,:) = dis;
            end
        end
    end
    
    if l == 1 
        weight_10242 = weight;
        save(strcat('E:\unc\zhengwang\dataset\format_dataset\weight_',num2str(vertices),'.mat'), 'weight_10242'); 
    elseif l ==2
        weight_2562 = weight;
        save(strcat('E:\unc\zhengwang\dataset\format_dataset\weight_',num2str(vertices),'.mat'), 'weight_2562'); 
    elseif l ==3
        weight_642 = weight;
        save(strcat('E:\unc\zhengwang\dataset\format_dataset\weight_',num2str(vertices),'.mat'), 'weight_642'); 
    elseif l ==4
        weight_162 = weight;
        save(strcat('E:\unc\zhengwang\dataset\format_dataset\weight_',num2str(vertices),'.mat'), 'weight_162'); 
    elseif l ==5
        weight_42 = weight;
        save(strcat('E:\unc\zhengwang\dataset\format_dataset\weight_',num2str(vertices),'.mat'), 'weight_42'); 
    end
    
end