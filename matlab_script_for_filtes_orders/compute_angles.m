function neighs_angle = compute_angles(neighs, center_pt)
    %project neighs to plane
    neighs_projected = zeros(size(neighs));
    for j = 1:length(neighs)
        t = 1 - (center_pt*neighs(j,:)')/(center_pt(1)^2 + center_pt(2)^2 + center_pt(3)^2);
        neighs_projected(j,:) = [neighs(j,1) + center_pt(1)*t, neighs(j,2) + center_pt(2)*t, neighs(j,3) + center_pt(3)*t];
    end
    
    neighs_angle = zeros(size(neighs, 1),1);
    if center_pt(1) ~= 0 || center_pt(2) ~= 0
        n_x = cross([0,0,1], center_pt); % normal of the new projected x-axis
        n_y = cross(center_pt, n_x); % normal of the new projected y-axis
    else
        n_x = [1,0,0];
        n_y = [0,1,0];
    end
    assert(abs(n_x * n_y') < 1e-6, 'n_x * n_y != 0');
    for j = 1:length(neighs)
        % caculate the angele between neighs and x-axis
        tmp = ((neighs_projected(j,:) - center_pt) * n_x')/norm(n_x)/norm(neighs_projected(j,:) - center_pt);
        if tmp > 1.0
            tmp = 1.0;
        elseif tmp < -1.0
            tmp = -1.0;
        end
        neighs_angle(j) =  acos(tmp);
        if (neighs_projected(j,:) - center_pt) * n_y' < 0
            neighs_angle(j) = 2*pi - neighs_angle(j);
        end
    end
end