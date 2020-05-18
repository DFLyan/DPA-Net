pics = dir('*.png')
num = length(pics)
cropnum = 1000
cropsize = 63
cropstart = 0
stride = 6
finnalpixel = stride*cropnum+1
all = 0

for i=1:num
    image = imread(strcat(num2str(i),'.png'));
    r = image(:,:,1);
    [n m] = size(r);
    new = 1;
    for j=1:stride:finnalpixel
        if (((n-cropstart-j)<cropsize)) == 1
            break
        else
            for k=1:stride:finnalpixel
                if (((m-cropstart-k)<cropsize)) == 1   %小于裁剪大小便跳至下一张图片
                    break
                else    
                    g = imcrop(image,[cropstart+k cropstart+j cropsize cropsize]);
                    imwrite(g,['4\',strcat(num2str(i),strcat('-',strcat(num2str(new),'.png')))]);
                    new = new+1;
                    all = all+1
                end       
            end
        end    
    end
end