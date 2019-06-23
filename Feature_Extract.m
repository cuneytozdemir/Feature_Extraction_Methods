clear all;
clc;
spham=0
pencereboyutu=4
spham=0;
secim="M"; % M= motif, A=angle, V=NeighborBasedOne-dimensionalLBP
myFolder = 'D:\Doktora\SpamDataset\lingspamPUBLİC';        
filePattern = fullfile(myFolder, '*.txt');
filelist = dir(filePattern);
features=[];
for a=1:length(filelist)
    baseFileName = filelist(a).name;
    spham=baseFileName(1:5);
    if (spham=="spmsg")
      spham=0;
    else
         spham=1;
    end
    fullFileName = fullfile(myFolder, baseFileName);    
    kk='';
    fid = fopen(fullFileName);
    tline = fgets(fid);
        while ischar(tline)
            kk=strcat(kk,tline);
            tline = fgets(fid);
        end
     fclose(fid);
if  (secim=="M")
     if(kk>1)
        bytes = unicode2native(kk) ; 
        bytes=bytes(bytes~=32);
        bytes=bytes(bytes~=253);
        bytes=bytes(bytes~=254);
        veri=bytes;
     end  
    [count]=motifyeni(veri,pencereboyutu); 
    
elseif (secim=="A")
   if(kk>1)
    bytes = unicode2native(kk) ; % 
    [a]=aci(double(bytes(1:end)));%     
    count=zeros(1,360);
     for k=1:360
     count(k)=length(find(a==(k-1)));
     end 
   end
else
  if(kk>1)     
    bytes = unicode2native(kk) ; 
    [a,b]=vectmap(double(bytes(1:end)),4,1,1);
    b=b';    
    count=zeros(1,256);
    for k=1:256
      count(k)=length(find(b==(k-1)));
    end
  end
end
    [count]=[count,spham];
     features=[features; count];
end    

csvwrite('C:\Users\aidata\Desktop\SpamDataset\lingMotif4.xlsx',features)
