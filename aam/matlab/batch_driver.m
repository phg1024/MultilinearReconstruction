clear;
visualize_results = false;

method = 'tournament';
repopath = '~/Data/InternetRecon3/%s';
persons = dir(sprintf(repopath, '*_*'));

%repopath = '~/Data/InternetRecon0/%s';
%person = 'yaoming';    
tic;
for i=4:length(persons)
    close all;
    person = persons(i).name
    if strcmp(person, 'Allen_Turing')
        continue;
    end
    AAMfilter(repopath, persons(i).name, method, visualize_results);
end
toc;