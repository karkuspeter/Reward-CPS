function [ params ] = ProcessParams( defaults,  input_params)
%overright params that are set in input_params

    params = defaults;
    fields = fieldnames(input_params);
    for i=1:numel(fields)
        if ~isfield(defaults, fields{i})
            fprintf('Warning: %s is not a parameter\n',fields{i});
        end
        params.(fields{i}) = input_params.(fields{i});
        
    end
%     fields = fieldnames(params);
%     for i=1:numel(fields)
%         if isfield(input_params, fields{i})
%             params.(fields{i}) = input_params.(fields{i});
%         end
%     end

end

