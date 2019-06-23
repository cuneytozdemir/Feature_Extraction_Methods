function M = indisBul(D)
    S=sort(D);
    M=[0,0];
    for i=1:length(D)
        M(i,1)=D(i);
        indx=find(S==D(i));
        for j=1:length(indx)
            if find(M(:,2)==indx(j))
                continue;
            else
                M(i,2)=indx(j);
                break;
            end
        end
    end
end
