%plot err

type = 1;

subplot(1,3,1);
for hops = [1,3,5,7]
    
    load(['TrainErrHopsNew-' num2str(hops) '.mat']);
    if TrainErr(type,11)==0
        plot(TrainErr(type,1:10),'LineWidth',1);
    else
        plot(TrainErr(type,1:20),'LineWidth',1);
    end
    hold on
    
end

type = 2;
subplot(1,3,2);
for hops = [1,3,5,7]
    
    load(['TrainErrHopsNew-' num2str(hops) '.mat']);
    if TrainErr(type,11)==0
        plot(TrainErr(type,1:10),'LineWidth',1);
    else
        plot(TrainErr(type,1:20),'LineWidth',1);
    end
    hold on
    
end

type = 3;
subplot(1,3,3);
for hops = [1,3,5,7]
    
    load(['TrainErrHopsNew-' num2str(hops) '.mat']);
    if TrainErr(type,11)==0
        plot(TrainErr(type,1:10),'LineWidth',1);
    else
        plot(TrainErr(type,1:20),'LineWidth',1);
    end
    hold on
    
end