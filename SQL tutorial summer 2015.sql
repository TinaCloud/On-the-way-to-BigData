# SQL-1
SHOW DATABASES;

USE dev_tt_799511;

SELECT database();

SHOW DATABASES; 
SELECT current_user();
SELECT database();

INSERT INTO Student(first_name,last_name,grade_level)
   VALUES('Fred','Doe',11),
         ('Lebron','James',12),
         ('Michael','Franklin',11),
         ('Robert','Stark',9);

CREATE TABLE Student
(
  student_id int(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY NOT NULL,
  first_name varchar(25) NOT NULL,
  last_name varchar(25) NOT NULL,
  grade_level tinyint(5) NOT NULL
);


DESCRIBE Student;

INSERT INTO Student(first_name,last_name,grade_level)
   VALUES('Fred','Doe',11),
         ('Lebron','James',12),
         ('Michael','Franklin',11),
         ('Robert','Stark',9);

SELECT * FROM Student;

SHOW TABLES;
DESCRIBE Student; -- Take a peek at the table.
SELECT first_name FROM Student;

SELECT * FROM Student WHERE first_name='Michael';

-- quiz

CREATE TABLE TestScore
(
  testscore_id int(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY NOT NULL,
  student_id int(10) UNSIGNED NOT NULL,
  test_name varchar(25) NOT NULL,
  score float(5, 2) NOT NULL,
  FOREIGN KEY (student_id)
        REFERENCES Student(student_id) -- Don't worry about this for now,
        --  but make sure Student table exists with student_id column.
);

INSERT INTO TestScore(student_id,test_name,score)
   VALUES(3,'AI Final 2',85.4),
         (1,'AI Final 2',78.1),
         (4,'AI Final 2',77.4),
         (4,'Bio Final 1',95.2);
         
SELECT * FROM TestScore;

# Destroy your existing tables:
SELECT database(); -- See what database we're using now.
DROP TABLE IF EXISTS TestScore; -- ** See below why we need to drop this.
DROP TABLE IF EXISTS Student;
DROP TABLE IF EXISTS Club;

# Create new tables:
CREATE TABLE Student
(
  student_id int(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY NOT NULL,
  first_name varchar(25) NOT NULL,
  last_name varchar(25) NOT NULL,
  grade_level tinyint(5) NOT NULL
);
 
CREATE TABLE Club
(
  club_id int(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY NOT NULL,
  student_id int(10) UNSIGNED NOT NULL,
  birthdate DATE NOT NULL,
  role varchar(25) NOT NULL,
  years_served tinyint(5) NOT NULL
  -- We're not going to include the foreign key clause here, so we don't have a hard constraint.
);

-- First Student
INSERT INTO Student(first_name, last_name, grade_level)
VALUES('Fred', 'Allen',9),
     ('Allen','Gates',12),
     ('Paula','Jobs',10),
     ('Benjamin','Franklin',10),
     ('Tony','Stark',10),
     ('Steve','Rogers',11),
     ('Marge','Simpson',12),
     ('Bart', 'Simplson',10)
;
 
-- Now Club
INSERT INTO Club(student_id, birthdate, role, years_served)
VALUES(1,'2001-07-09','greeter',1), -- For date use YYYY-MM-DD
     (2, '1997-05-22','president',4),
     (3,'2000-09-17','treasurer',2),
     (4,'2000-02-14','vice president',2),
     (9, '2001-12-14', 'member',1),
     (10,'1998-10-10','member',1),
     (11,'1997-01-06','secretary',3),
     (12,'2000-02-13','member',3)
 
;

SELECT * FROM Club;
SELECT * FROM Student;

SELECT first_name,last_name FROM Student
INNER JOIN Club
ON Student.student_id = Club.student_id;

SELECT birthdate FROM Student
INNER JOIN Club
ON Student.student_id = Club.student_id;

SELECT * FROM Student
INNER JOIN Club
ON Student.student_id = Club.student_id
WHERE years_served > 1; 

SELECT * FROM Student
LEFT OUTER JOIN Club
ON Student.student_id=Club.student_id;

SELECT * FROM Student
RIGHT OUTER JOIN Club
On Student.student_id=Club.student_id;

SELECT * FROM Shared.Movie
INNER JOIN Shared.Rating
On Shared.Movie.movie_id=Shared.Rating.movie_id;

USE Shared;
DESCRIBE Shared.Movie;

-- SQL-2
-- Create the table that you will import into.
DROP TABLE MyTag;

CREATE TABLE MyTag
(
  rating_id   int(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY NOT NULL,
  user_id     int(10) UNSIGNED NOT NULL,
  movie_id    int(10) UNSIGNED NOT NULL,
  tag         varchar(128) NOT NULL,
  time_tagged int(11) NOT NULL
);

-- Update the path and execute the query.
LOAD DATA LOCAL INFILE 'e:\\tags.csv' -- this string is where the file is on your computer.
INTO TABLE MyTag                       -- Name of the table you created earlier
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'             -- You want to change this to '\r' if you're on a MAC!
IGNORE 1 LINES                         -- This is for the header
(user_id, movie_id, tag, time_tagged); -- name of the matching columns

DESCRIBE MyTag;

SELECT COUNT(*) FROM MyTag;

USE Shared;
SHOW TABLES;

DESCRIBE Shared.Rating;

SHOW Databases;
-- Make sure you have the Shared database.
SELECT sum(rating) FROM Shared.Rating;

SELECT COUNT(*) FROM Shared.Rating;

DESCRIBE Shared.Movie;

SELECT * FROM Shared.Movie WHERE title='GoldenEye';

SELECT AVG(rating) FROM Shared.Rating WHERE movie_id=10;

SELECT min(year_released) FROM Shared.Movie;

DESCRIBE Shared.Rating; -- First let's see what's in the table.
 
-- Now for the aggregation.
SELECT movie_id, AVG(rating) FROM Shared.Rating GROUP BY movie_id;

SELECT  user_id, count(rating) FROM Shared.Rating GROUP BY user_id;

SELECT user_id, AVG(rating) FROM Shared.Rating GROUP BY user_id;
-- We know that the following will yield the average rating for all movies.
SELECT COUNT(Rating) from Shared.Rating;
 
SELECT movie_id FROM  Shared.Rating
WHERE rating < (SELECT AVG(rating) from Shared.Rating);

SELECT AVG(rating) FROM Shared.Rating 
WHERE rating < (SELECT AVG(rating) from Shared.Rating)
;

DESCRIBE Shared.Movie;
 
DESCRIBE Shared.Tag;
 
DESCRIBE Shared.Rating;


