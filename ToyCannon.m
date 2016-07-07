classdef ToyCannon
    % Class for toy cannon problem
    %   2D cannon with random hills
    
    properties
        angleNoise
        s_bounds
        hill
        x  %positions on ground, cached
        y  %hill values for each x
        r_func
        PrintOn;
    end
    
    methods
      function obj = ToyCannon()
         obj.angleNoise = 1/180*pi;
         obj.s_bounds = [0, 12];
         obj.hill = struct('c', [3 5 6 7], ...
                           'h', [.3 .3 .4 .2], ...
                           'scale', [.5 .5 .1 2]);
         obj.x = obj.s_bounds(1):0.1:obj.s_bounds(2);
         obj.y = obj.HillValue(obj.x);
         obj.r_func = @(a,v,s,hillats,xres,yres)(4-sqrt( (xres-s).^2 + (yres - hillats)^2));
         % reward: eucladian distance from target on the hill,
         % +4 to address 0 mean
         % could be x difference only, and add penalty for higher angle?
         
         obj.PrintOn = false;
      end

      function PlotEnv(obj)
          x = obj.s_bounds(1):0.1:obj.s_bounds(2);
          plot(x, obj.HillValue(x))
      end
      
      function y = HillValue(obj, x)
          y  = zeros(size(x));  %hill values
          for i = 1:length(obj.hill.c)
              y = y + obj.hill.h(i) * exp(-(x-obj.hill.c(i)).^2/obj.hill.scale(i));
          end
      end
      
      function [r, xres] = Simulate(obj, s, angle, v, noise)
          % reasonable input: s=3, angle=.3, v = 1;
          if nargin < 5
              noise = obj.angleNoise;
          end
          
          th = angle + randn(1)*noise;
          
          yproj = obj.x.*tan(th) - 9.81/100 * obj.x.^2/2/(v*cos(th))^2; %y of ball trajectory
          
          ixLand = find(bsxfun(@lt, yproj, obj.y));  % find all x where yproj < y
          ixLand = ixLand(2); % ixLand(1) is the initial pos, ixLand(2) is where it hits the hill
          xres = obj.x(ixLand);
          yres = obj.y(ixLand);
          
          %evaluate
          hillats = obj.HillValue(s);
          r = obj.r_func(angle,v,s,hillats,xres,yres);
          
          %print
          if(obj.PrintOn)
              figure(1);
              PlotEnv();
              hold on, plot(obj.x(1:ixLand), yproj(1:ixLand), 'r--')
              hold on, plot(s, ycontext, 'ko')
          end
      end
      
    end
    
end