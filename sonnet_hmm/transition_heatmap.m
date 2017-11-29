A = dlmread('couplet_transition.txt');

colormap copper
imagesc([1, 30],[1, 30], flipud(A)) 
colorbar

title('A_{transition} for Couplets')
xlabel('State y^{j + 1}')
ylabel('State y^{j}')

x = 200; 
y = 200; 
width = 600; 
height = 500; 
set(figure(1), 'Position', [x y width height])
a = findobj(gcf); 
allaxes = findall(a, 'Type', 'axes'); 
alllines = findall(a, 'Type', 'line'); 
set(alllines, 'Linewidth', 1); 
set(allaxes, 'FontName', 'Helvetica', 'FontWeight', 'Bold', 'LineWidth', 2, ...
    'FontSize', 20, 'box', 'on')

figure(2)
hold on

A_start = dlmread('couplet_start.txt');

colormap copper
resol = 1; 
imagesc([1, 30],[1, 1], flipud(A_start)) 
colorbar

title('A_{start}')
ylabel('State y^{j}')
ylim([1, 30])
xlim([0.9999, 1])
set(gca,'XTick',[])


x = 200; 
y = 200; 
width = 200; 
height = 500; 
set(figure(2), 'Position', [x y width height])
a = findobj(gcf); 
allaxes = findall(a, 'Type', 'axes'); 
alllines = findall(a, 'Type', 'line'); 
set(alllines, 'Linewidth', 1); 
set(allaxes, 'FontName', 'Helvetica', 'FontWeight', 'Bold', 'LineWidth', 2, ...
    'FontSize', 20, 'box', 'on')