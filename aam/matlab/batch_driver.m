clear;
visualize_results = false;

method = 'tournament';
repopath = '~/Data/InternetRecon3/%s';
persons = dir(sprintf(repopath, '*_*'));

%repopath = '~/Data/InternetRecon0/%s';
%person = 'yaoming';    
tic;
for i=4:length(persons)
    person = persons(i).name
    if strcmp(person, 'Allen_Turing')
        continue;
    end
    driver
end
toc;