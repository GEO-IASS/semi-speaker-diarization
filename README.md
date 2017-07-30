- Input-Output parameters and their descriptions/specifications are provided in each function file 

- semi_supervised_find.m is a function file to spot a speaker
- semi_supervised_re_iter.m is a function file to re-iterate the speaker spotting process
- error_calc_new.m calculates error estimates 
- model_update.m updates the speaker models using frames of data provided

- Run demo.m, user needs to input values for variables 'chosen_start' and 'chosen_end'. Program displays error  
  estimate parameters (true-positive, false positive and false negative) values at the end of every iteration of speaker spotting process along with label of the speaker being spotted.
