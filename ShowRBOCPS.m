    if (theta_dim > 1)
        context = context_vec(context_id);
        [~, Rstar] = MapToContext(Dfull, context, @problem.r_func, isPopulate, 0, st_bounds);

        [x1, x2] = ndgrid(linspace(theta_bounds(1,1),theta_bounds(1,2), 100), ...
            linspace(theta_bounds(2,1),theta_bounds(2,2), 100));
        %y = arrayfun(@(t1, t2)(problem.sim_eval_func(context, [t1 t2])), x1, x2);
        
        y = problem.get_cached_grid(context_id, 100, 100);
        
        %[r_opt, ind] = max(y(:)); %will return 1d indexing
        %theta_opt = [t1(ind), t2(ind)];
        %current_and_opt = [[theta_vec(context_id, :) r_vec(context_id, :)]; ...
        %theta_opt r_opt]
               
    else
        
        [x1, x2] = ndgrid(linspace(bounds(1,1),bounds(1,2), 100), ...
            linspace(bounds(2,1),bounds(2,2), 100));
        y = arrayfun(@problem.sim_eval_func, x1, x2);
    end
    
        figure(1);
    mesh(x1, x2, y);
    hold on
    if theta_dim == 1
        plot3(context_vec(:,1), theta_vec(:,1), r_vec);
        plot3(x1, theta_opt, r_opt);
    else
        scatter3(Dfull.theta(:,1), Dfull.theta(:,2), Rstar);
        scatter3(theta_opt(context_id,1), theta_opt(context_id,2), r_opt(context_id),'*' );
        scatter3(theta_vec(context_id,1), theta_vec(context_id,2), policy_pred_vec(context_id),'*' )        
    end
    hold off
    xlabel('context');
    ylabel('angle');
    colorscale = caxis();
    view(0,90)
    %legend('Real R values');
    
    %if (~params.EvalAllTheta)
    %    drawnow;
    %    continue;
    %end
    % show prediction
    %[x1, x2] = meshgrid(linspace(bounds(1,1),bounds(1,2), 100), ...
    %    linspace(bounds(2,1),bounds(2,2), 100));
    %Xplot = reshape(cat(2, x1, x2), [], 2);
    %[ypred, ystd] = predict(gprMdl, Xplot);
    if theta_dim == 1
        Yplot = reshape(ypred, size(x1,1), []);
    else
        Yplot = reshape(ypred, [], size(context_vec,1)); %each column is a context
        Yplot = Yplot(:, context_id);
        Yplot = reshape(Yplot, 100, 100);
    end
    
    figure(2);
    mesh(x1, x2, Yplot);
    hold on
    if theta_dim == 1
        scatter3(Dfull.st, Dfull.theta, Dfull.r);
    else
        scatter3(Dfull.theta(:,1), Dfull.theta(:,2), Rstar);
        scatter3(theta_vec(context_id,1), theta_vec(context_id,2), policy_pred_vec(context_id),'*' );
    end
    xlabel('context');
    ylabel('angle');
    caxis(colorscale);
    %legend('Data','GPR predictions');
    view(0,90)
    hold off
    
    
    if theta_dim == 1
        Yplot = reshape(ystd, size(x1,1), []);
    else
        Yplot = reshape(ystd, [], size(context_vec,1)); %each column is a context
        Yplot = Yplot(:, context_id);
        Yplot = reshape(Yplot, 100, 100);
    end
    figure(3);
    mesh(x1, x2, Yplot);
    xlabel('context');
    ylabel('angle');
    %view(0,90)
    hold off
    
    if theta_dim == 1
        Yplot = reshape(acq_val, size(x1,1), []);
    else
        Yplot = reshape(acq_val, [], size(context_vec,1)); %each column is a context
        Yplot = Yplot(:, context_id);
        Yplot = reshape(Yplot, 100, 100);
    end
    
    figure(4);
    mesh(x1, x2, Yplot);
    hold on
    if theta_dim == 1
        scatter3(Dfull.st, Dfull.theta, Dfull.r);
    else
        scatter3(Dfull.theta(:,1), Dfull.theta(:,2), Rstar);
    end
    xlabel('context');
    ylabel('angle');
    %legend('Data','Aquisition function');
    view(0,90)
    hold off
    limits = axis();

    figure(5);
    scatter3(Dfull.theta(params.InitialSamples+1:end,1), Dfull.theta(params.InitialSamples+1:end,2), Rstar(params.InitialSamples+1:end));
    axis(limits);

    figure(6);
    if theta_dim == 1
    else
        plot3(theta_opt(:,1), theta_opt(:,2), context_vec(:,1));
        hold on
        scatter3(theta_opt(:,1), theta_opt(:,2), context_vec(:,1), '*');

        plot3(theta_vec(:,1), theta_vec(:,2), context_vec(:,1));
        scatter3(theta_vec(:,1), theta_vec(:,2), context_vec(:,1));
    end
    hold off
    xlabel('theta_a'); ylabel('theta_v'); zlabel('context');
    
    %figure(7);
    %problem.PlotEnv();
    
    
    drawnow
    %pause;