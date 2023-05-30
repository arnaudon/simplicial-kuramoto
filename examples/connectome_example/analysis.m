

%% Reproduce Figure 11a
cm = gray;
colormap(flipud(max(min(cm+[0,0.1,0.1],1),0)))
sc = load('KSmodel_fMRIdynamics-main/sch200_SC.mat').Kij;
imagesc(log10(sc))
%clim([-0.75 0.75])
xticks([''])
yticks([''])
colorbar
daspect([1 1 1])

%% Reproduce Figure 11b
m = gray;
colormap(max(min(cm+[0.08,0,0],1),0))
imagesc(fc)
xticks([''])
yticks([''])
clim([-0.75 0.75])
daspect([1 1 1])
colorbar

%% Reproduce Figure 11c
tiledlayout(1,5,'TileSpacing','compact')
%colormap(flipud(lbmap(256,'RedBlue')))
cm = gray;
colormap(max(min(cm+[0.08,0,0],1),0))
for i = 1:5
    nexttile
    imagesc(squeeze(matrices(i,1,:,:)))
    xticks([''])
    yticks([''])
    clim([-0.75 0.75])
    daspect([1 1 1])
end
colorbar


%% Reproduce Figure 11d

err_fc = load("correlations_final/model_correlations.mat").corrs2;
err_fc = err_fc(1:5,:);
sigma_space = linspace(100,500,20);
plot(sigma_space,err_fc','-')

ylabel('Pearson Correlation','Interpreter','latex','FontSize',16)
xlabel('$\sigma$','Interpreter','latex','FontSize',16)
[~,idx] = max(err_fc');
hold on
scatter(sigma_space(idx),diag(err_fc(:,idx)),'filled')
legend(["Node OI","Edge","Edge OI","Node OM","Edge OM",""],'Interpreter','latex')


%% Reproduce Figure 11e

sc = load('KSmodel_fMRIdynamics-main/sch200_SC.mat').Kij;
fc = load('KSmodel_fMRIdynamics-main/sch200_FC.mat').nFCavg;
matrices = zeros(6,10,200,200);
matrices(1,:,:,:) = load("matrices_final/node_FC.mat").matrices;
matrices(2,:,:,:) = load("matrices_final/edge_FC.mat").matrices;
matrices(3,:,:,:) = load("matrices_final/edgeOI_FC.mat").matrices;
matrices(4,:,:,:) = load("matrices_final/nodeOM_FC.mat").matrices;
matrices(5,:,:,:) = load("matrices_final/edgeOM_FC.mat").matrices;


corrs = zeros(5,10);
for i = 1:5
    for j = 1:10
        corrs(i,j) = fc_dist(squeeze(matrices(i,j,:,:)),fc);
    end
end
corrs = corrs';

hold on
plot(mean(corrs),'--o','LineWidth',1,'Color',[0.4,0.4,0.4])

for i = 1:5
    boxchart(repmat(i,10,1),corrs(:,i))
end

set(gca,'FontSize',14) 
set(gca,'TickLabelInterpreter','latex');
xticks([1 2 3 4 5])
xticklabels({'Node','Edge','Edge OI','Node OM','Edge OM','Edge OM OI'})
xtickangle(45)
xlim([0.5 5.5])
ylabel("Pearson Correlation",'Interpreter','latex')

%% Test differences in correlation averages
gnames = ["NodeOI","Edge","EdgeOI","NodeOM","EdgeOM"];
tab = table(corrs(:,1),corrs(:,2),corrs(:,3),corrs(:,4),corrs(:,5),'VariableNames',gnames);
tab = stack(tab,gnames,'NewDataVariable',"corr",'IndexVariableName',"model");
[p,t,stats] = anova1(table2array(tab(:,2)),table2array(tab(:,1)));
[results,~,~,gnames] = multcompare(stats);
tbl = array2table(results,"VariableNames", ...
    ["Group","Control Group","Lower Limit","Difference","Upper Limit","P-value"]);
tbl.("Group") = gnames(tbl.("Group"));
tbl.("Control Group") = gnames(tbl.("Control Group"))

%% Compute Effect Sizes
meanEffectSize(corrs(:,2),corrs(:,1))
meanEffectSize(corrs(:,5),corrs(:,1))

