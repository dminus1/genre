Natural annotation means that we deal with certain text classes which
naturally prevalent in a given repository. More specifically, for genres
we can rely on the fact that certain genres are more natural in certain
sources, for example, regular Wikipedia articles mostly provide
reference information.

This repository uses the following classes and the following sources

| Genre       | General prototypes              | \#Texts | Natural source | Topical bias     |
|-------------|---------------------------------|---------|----------------|------------------|
| ARGument    | Expressing opinions, editorials | 126755  | Hyperpartisan  | Topics 9, 13     |
| INSTRuction | Tutorials, FAQs, manuals        | 127472  | StackExchange  | Topics 19, 21    |
| NEWS        | Reporting newswires             | 16389   | Giga News      | Topics 5, 9      |
| PERSonal    | Diary entries, travel blogs     | 16432   | ICWSM'09 set   | Topic 23         |
| INFOrmation | Encyclopedic articles           | 97575   | Wikipedia      | Topics 1, 15, 20 |
| Review      | Product reviews                 | 1302495 | Amazon reviews | Topics 1, 16, 17 |
|             | **Total**                       | 1687118 |                |                  |

Keywords from ukWac for the **topic model** with 25 topics:

| Label: Nr          | Top keywords                                                                                         |
|--------------------|------------------------------------------------------------------------------------------------------|
| Finances: 0        | insurance, property, pay, credit, home, money, card, order, payment, make, tax, cost, time           |
| Entertain: 1       | music, film, band, show, album, theatre, festival, play, live, sound, radio, song, dance, songs      |
| Geography: 2       | road, london, centre, transport, park, area, street, station, car, north, east, city, west, south    |
| Business: 3        | business, management, company, service, customers, development, team, experience                     |
| University: 4      | students, university, research, learning, skills, education, training, teaching, study, work         |
| Markets: 5         | year, market, million, energy, waste, years, cent, industry, investment, government, financial       |
| Web: 6             | information, site, web, website, page, online, search, email, click, internet, details, links, find  |
| Science: 7         | data, research, system, analysis, model, results, number, time, science, methods, surface, cell      |
| $\*$Webcleaning: 8 | 2006, 2005, posted, 2004, june, july, october, march, april, september, 2003, august, january        |
| Politics1: 9       | government, world, people, international, war, party, countries, political, european, labour         |
| Travel: 10         | hotel, room, day, area, house, accommodation, holiday, visit, city, centre, facilities, town, great  |
| Health: 11         | health, patients, treatment, care, medical, hospital, clinical, disease, cancer, patient, nhs, risk  |
| Councils: 12       | development, local, community, council, project, services, public, national, planning, work          |
| Life1: 13          | people, time, questions, work, make, important, question, problem, change, good, problems            |
| Software: 14       | software, system, file, computer, data, user, windows, digital, set, files, server, users, pc, video |
| Sports: 15         | game, club, team, games, play, race, players, time, season, back, football, win, world, poker        |
| Religion: 16       | god, life, church, people, lord, world, man, jesus, christian, time, love, day, great, death, faith  |
| Arts: 17           | book, art, history, published, work, collection, world, library, author, london, museum, review      |
| Law: 18            | law, act, legal, court, information, case, made, public, order, safety, section, rights, regulations |
| Nature: 19         | food, water, species, fish, plants, garden, plant, animals, animal, birds, small, dogs, dog, tree    |
| History: 20        | years, century, house, st, john, royal, family, early, war, time, built, church, building, william   |
| Engineering: 21    | range, design, light, front, high, car, made, water, power, colour, quality, designed, price         |
| Politics2: 22      | members, meeting, mr, committee, conference, year, group, event, scottish, council, member           |
| Life2: 23          | time, back, good, people, day, things, make, bit, thing, big, lot, can, long, night, feel, thought   |
| School: 24         | people, children, school, support, young, work, schools, child, community, education, parents        |
|                    |                                                                                                      |

The Amazon reviews are from
<http://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz>

The full sample is at
<http://corpus.leeds.ac.uk/serge/webgenres/natural/full-sample.tar.xz>
