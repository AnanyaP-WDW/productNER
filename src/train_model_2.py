# ;==========================================
# ; Author: Ananya Pathak
# ; Date: May 12, 2021
# ;==========================================

import spacy
from spacy import displacy
from spacy.matcher import Matcher
import re
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example

def parse_train_data(doc):

    measure_detections = list()
    start_for_product_name = matcher(doc)[0][1]
    #print(start_for_product_name)
    product_name_detections = [(doc[:start_for_product_name].start_char, doc[:start_for_product_name].end_char, 'product_name')]
    #print(product_name_detections)

    measure_detections.extend(product_name_detections)
        
    return (doc.text, {'entities': measure_detections})

def clean_text(input_text):
    input_text =  input_text.strip().lower()
    
    input_text = re.sub(r"[\n\t<>()/]", " ", input_text)
    input_text = " ".join(input_text.split())
    
    return input_text


if __name__ == '__main__':
    input_string = """cantata (smash)hazelnut incense coffee beans 600g/mocha blend
    Costa costa signature blend coffee bean 200G x 2
    starbucks veranda blend light smash coffee 567g 2 pack
    trader joe organic french roast 13oz(369g) x2 pack
    [blackvins]mandeling, indonesia coffee beans after ordering roasting 200g
    for restaurant prima vending machine cohabitation 1kg coffee prim business
    [117557]ethiopia special blend(500g/fine grinding/mr bin)
    [keopimandeulgi] kenya aa TOP 500g
    rabbah intensos dark roast coffee beans powder 340g 3 piece
    pauls ground coffee beans 865g coffee powder drip coffee
    pauls breakfast blend mild light roast ground coffee 720g
    indonesia sumatra mandeling 100g coffee beans roasting on the day
    [sales cafe] hazelnut incense coffee beans(200g/500g/1kg)
    espresso bold dark 1kg(SRBC140FC0500A02)coffee beans
    el salvador fancy 1kg(SROC120CI0500A02) gobgebunswae/gajeongyongeseupeureso/mocha pot(M)
    [coffee pilgrims] guatemala shb beans coffee/227g 1 rod(8oz 1 rod) /single origin
    brazil serado coffee beans 1kg roasting on the day
    sanjibyeolsaengdu santos, brazil NY2 1Bag(60kg)
    ilya classico holvin coffee medium roast 250g 6 pack
    The Coffee Fool teugjeon coffee fool english topi 12oz
    antigua guatemala 1kg guatemala coffee beans roseutingweondukeopi coffee beans coffee bean gwatemalr
    brazil seahawo 1kg /bean(holvin)
    Tim Hortons hazelnut light geuraindeukeopi 300g 2 pack
    constantine 500g 100% arabica hazelnut incense ground beans coffee
    davidoff rich aroma coffee 100g 1 bottle.
    carmonnet blueberry latte 500g
    (radish)maxwell original(for vending machines)900gX12 piece
    maxwell house coffee mix original 11.8g X 100T
    G7 G7 coffee 3in1 mixed coffee 16g x 100 intervention for export (100T + 10T)
    east and west maxim keopiman(black) mocha gold mild (20T)
    east-west foods maxwell hwainkeopi 500g dongseosigpumkeopi dongseokeopi maxwell house
    maxim mocha gold refill 500g coffee stick disposable coffee stick coffee husigkeopi dijeoteukeopi
    [mokona] instant coffee 100g midieomroseuteu
    stationery office/nobeulkeopikeopimigseu 100T(1gx100T/ildong foods)
    ttt maxim white gold coffee mix(11.7gX180T dongseo food)
    [G7] G7 vietnam black coffee mix 15T
    G7 instant coffee 30g X 5 coffee,mix,bean,bongjikeo
    maxim mocha gold mild coffee mix 12g x 800 piece
    nescafe coffee 200g
    for vending machines coffee mix food health beverage japangiyongkeopimigseu 900gx12 bag coffee maxwe
    vietnam G7 coffee 3in1 mixed coffee domestic use 800g(16g x 50T)
    east and west maxim original coffee 100g (bottle) 12 intervention box
    davidoff cafe rich aroma stick coffee 1.8g x25 intervention
    [starbucks] double shot espresso&amp;200 ml cream 36 mouth
    lotte cantata cold brew black 275ml 20 cans
    georgia cafe latte 240ML X 30 cans
    georgia ice black 220ml x50 piece GG8230
    tiffany master latte 200ml x 30 piece/free shipping
    [guangdong] kapedeurobtabkaperadde 275ml x24 piece /coffee/drink
    [guangdong pharmaceutical] [guangdong] kapedeurobtabkaperadde 275ml x24 piece /coffee/drink
    lotte cantata premium latte 275ml 20 cans
    cantata karamelmakiato&americano 275mlXgag 10(gun 20 piece)
    tiffany latte&cantata ame 275ml X bracket 12 piece(gun 24 piece)
    [haen food] cafe bene cafe latte 200ml x 10 things
    TOP tiffany simpeulriseumuseu roseutibeulraeg 360ml x 20 pet / coffee drink
    TOP tiffany simpeulriseumuseu roseutiradde 360ml x 20 pet / coffee drink
    costco ratsby mild cans of coffee 190ml x 30
    lotte chilsung lesbian cafe time 250ml x 30 cans
    maxim TOP master latte [275mlx20 cans]
    french cafe 200ml cup 10 intervention cup coffee 8 bell coffee drink
    tiffany simpeulriseumuseu roseuti black 360ml x 20 pet
    tioffy suites americano 275ml (20 cans) tiopihanbagseu
    L'OR Instant Coffee Classique FD - 100g (0.22lbs)
    Caffe D’Vita Hazelnut Cappuccino 1 lb can (16 oz)
    Kicking Horse Coffee, Three Sisters, Medium Roast, Ground, 284 g - Certified Organic, Fairtrade, Kos
    Union Hand Roasted Organic Brazil Filter Coffee (227g)
    U C C class one bagged 150g
    nescafe gold blend fragrant and gorgeous(120g)[(nescafe(NESCAFE))] 
    free shipping S A L E special limited price nescafe gold blend caffeine-less eco&system pack 6 0 gx5
    nescafe tesutazuchiyoisu de cafe one 7 5 g one 0 this set  free shipping nationwide
    A G F M A X I M bottle eight 0 gx24 piece
    U C C varistor espresso roast one 5 5 g 1 box(30 canned)
    Kenco Decaffeinated Instant Coffee 100G
    NESCAFÉ GOLD Instant Coffee Almond Latte (16g Sachets, Pack of 30)
    free shipping daido  blend BLACK(black) world one supervised by barista 2 7 5 g bottle canx24 book *
    free shipping [(2 case set)] giraffe koiwai milk and coffee 5 0 0 m l pet bottlex24 bookx(2 cases) *
    DyDo dido drinko daido  blend micro sugar world one supervised by barista to the last continue rich 
    U C C blended coffee regular one eight 5 g can x 3 0 book case sale (coffee beverage)
    coca・coke georgia emerald mountain blend blissful micro sugar one eight 5 g can 3 0 piece
    sangalia regular latte one 9 0 g canx30 pieces[Shang Wei Qi Xian :4 months or more]hokkaido, okinawa
    ziyoziaemerarudomauntenburendo one eight 5 g can 5 cans packxsix[(total 3 0 can)] 
    georgia fragrance black 4 0 0 m l bottle canx2 four coca cola products
    asahi wanda black sugar-free one eight 5 gx30 cans
    free shipping pokka sapporo drinking milk cafe latte 5 0 0 m l pet bottlex24 book *hokkaido・okinawa 
    U C C ueshima coffee T H E D R I P(the drip) roasted deep iced coffee sugar-free one 0 0 0 m l 1 box
    A G F burendeibotorukohi sugar-free 9 0 0 m lx12 piecesx4 cases (48 pieces) beverage[(free shipping*
    georgia deep black one eight 5 g canx30 pieces(1 case)
    sangalia quality coffee charcoaled (185gx30 pieces)x3 boxes
    suntory coffee craft boss latte(5 0 0 m lx24 piece)x2 cases
    can coffee kirin fire direct flame blend 1 eight 5 g canx3 0 book 1 case
    georgia fragrance black 4 0 0 m l bottle canx24 piece
    PRIMA nicety ground coffee 100 g
    lavazza super crema 1kg coffee beans f vat
    mk coffee cafe brazil grain 250g 981 8
    coffee astra gentle crema arabica 1kg grain
    coffee brazil santos 2kg freshly roasted arabika100
    Movenpick Der Himmlische 500g x2 coffee beans
    coffee beans ideas Caffe Crema Milder 1kg
    tchibo coffee beans Exclusive 2 kgs
    Jacobs KRONUNG without caffeine Eintkoffeiniert 500g
    Najjar 100% Arabica Coffee with Ground Cardamom 200g
    Herbal Arabica Coffee with Maitake Organic 250g Refill Pack with Vanilla
    Herbal Arabica Coffee with Wild Lettuce 250g Refill Pack
    Herbal Arabica Coffee with Carob Organic 250g Refill Pack with Cinnamon
    Coffee World | Brazil Yellow Bourbon Single Origin Arabica UK Roasted Whole Coffee Beans - Perfect B
    Coffee World | Colombia Excelso Rainforest Alliance Single Origin Arabica UK Roasted Whole Coffee Be
    Taylors of Harrogate Decaffé Ground Coffee
    degreaser Sonett, 120ml
    ground coffee natural strength
    brown coffee with taste of orange, express, 200 g
    ground coffee SOLEY NATURAL 250 G
    naturata organic cereal coffee, instant (1 x 200 gr)
    Mount Hagen Instant Fair Trade, 6 pack (6 x 100 g) - Bio
    coffee coffee beans coffee coffee beans trial coffee powder powder beans the sun's kingdom is brazil
    key coffee/F P ground taste mellow mild blend(powder)330g
    mocha・irugachiehu(raw beans)[200g]
    avance AVANCE appraiser recommendation special blend (7.5gx18P)135g
    Nescafe Gold Blend coffee, weight 750 g
    L'gold, classical refill, 180g
    Kopiko Macchiato Coffee 168g. 7sachets
    Nescafe Red Cup Instant Coffee 180 g.
    Simon levelt Organic Ethiopia Ground Coffee 250g
    caramel flavored filter coffee core 1 kg
    Colombia Altamira Excelso Single Origin local filter coffee 250 gr
    Lavazza Dek Intenso coffee 250g
    moxxa - Crema 250 g ground
    Brown Bear Sweet Brazil bright/Medium coffee beans 1 kg
    1kg - fresh roasted coffee - coffee shop Cult - indonesia - Java - West Blue - whole beans
    Corsini Single El Salvador Strictly High Grown GemahlenerKaffe 250g
    GrosshandelPL Woseba Rodzinna ground coffee 24x250g
    GrosshandelPL Jacobs Barista Smooth ground coffee pack of 20 (20x400g)
    Dallmayr - coffee Prodomo decaffeinated (Cafe moulu) | totally weight 500 grams
    bourbon coffee beans beans toasted red mixture Red 2Kg
    Dunkin Donuts Bakery series cinnamon flavored coffee roll Ground Coffee - American imported roast Ka
    UCC(you shishi)117 instant coffee 90 g/bottle
    Cafe Rey Tarrazu Ground Coffee, Costa Rica, 500 g/17.06 oz.
    ground coffee deca - 500 g
    Cafe INDUBAN Gourmet Tostado en Grano / Ganze Bohnen Premium Kaffee aus der Dominikanischen Republik
    grandmother coffee tasting ground 500 g - lot of 3
    Lavazza Original Qualita Rossa Espresso Coffee 250g
    great rice brown venere rice - 1kg
    NESCAFE GOLD ESPRESSO soluble coffee jar 100g
    Kenco Rich Instant Coffee echo Refill 150 g (Pack of 6)
    zenulife 100% Pure And Natural Green Coffee Beans Powder - 800 Gm (Pack Of 3) Instant Coffee
    Vihado 100% Pure And Organic Green Coffee Beans Powder - 200g (Pack Of 6) Instant Coffee
    Zenulife Organic Green Coffee beans for Weight Management 250 GM Instant Coffee pack of 4 Instant Co
    coffee lavazza Crema e Gusto Classico ground 250 g
    coffee Vergnano Gran Aroma in grains in vacuum packing 1 kg
    100% Arabica Coffee Beans 200gm Pouch by Dukens DUKACB200
    Tassyam Roasted Arabica Coffee Beans
    instant coffee JARDIN Colombia Medellin sublimated, 150g
    gourmet coffee ground the farm european toasted 340 g
    GOOD DAY AVOCADO DELIGTH COFFEE 250 ML
    JJ Royal Java Gold Blend Ground Bag [200 gr]
    aroma coffee Robusta Bandung [fine milled/ 250 gr]
    Jacobs Kronung Instant Coffee Refill 150g
    Nescafe Tasters Choice decaffeinated 48 gr Nescaf
    ground coffee Bonafide sensations Torrado Intenso 250 Gr
    roasted coffee ground FARACO, package 500 grams
    ground coffee extra strong 500g Melitta
    coffee powder traditional Melitta 250g vacuum Kit 3un
    ilya nespresso hohwankaebsyulkeopi lungo 10 intervention x 3
    ilya nespresso hohwankaebsyulkeopi 10 intervention 4 set of bells
    starbucks neseupeuresohohwan capsule coffee 80 capsule pick
    [dalmaieo] kaebsa capsule coffee espresso barista 10 capsule (nespresso capsule machine compatible /
    rabbah nespresso compatible capsules dikapeinato 1 paegx10 capsule
    Nespresso iseupiracione florence arpeggio 50 capsule
    [foolish love]rabbah dolce gusto compatible capsule coffee keuremoso 16 intervention
    tambisil georgia gotica espresso aegsangseutig 120 piece
    rabbah gran selrejione dark roast coffee capsule 60 piece
    Peets capsule variety pack nespresso compatible 10 capsule 4 pack
    Gevalia gebalria caramel makati cup 6 intervention 6 pack
    ilya coffee machine decaffeine medium roast capsule 21 piece
    pauls vanilla biscotti queue league k cup capsule coffee 96 piece
    Green Mountain capsule coffee French Vanilla 72 intervention
    community coffee New Orleans Blend Dark Roast 12 capsule
    nespresso orijineolrain intensos yeoreogajimas 50 piece
    nescafe dolce gusto seukiniradde makiato 3X16 48 capsule
    Coffee Bean & Tea Leaf CBTL eseupeureseu capsule 16 intervention
    Starbucks starbucks capsule coffee columbia 10 capsule 5 pack
    rabbah peopeto keikeob queue league espresso 16 piece 1 pack
    T dolce gusto americano maegsipaeg 10gX48 mouth coffee capsule
    nespresso Stormio Odacio Melozio 30 capsules
    ilkapeitalriano NEW venice 80 intervention capsule coffee
    rabbah lungo aboljente dark roast capsule keompaeteobeul 60 pack
    ilya iperEspresso decaffeine coffee capsule 21 intervention
    80 capsules Nocciolino ocelot compatible nespresso
    nescafe sweet taste Cappuccino 16 Capsule (8 cups)
    40 capsule pods compatible nespresso refreshes CAFFE GINSENG RESPRESSO original
    40 capsules compatible caffitaly blend lovely
    400 capsules gattopardo coffee decaffeinated compatible nespresso
    a system 200 capsules coffee rich taste ocelot
    bourbon coffee 300 capsules FAP lavazza espresso compatible point mixture DEK decaffeinated - CAFFE'
    senseo coffee pods Premium Set classic/Classic, 3 pack, intensive & full-bodied taste, coffee, ever 
    Nescafe Dolce Gusto Zoegas Skanerost, coffee capsule, coffee, 16 capsules (16 servings)
    Lavazza Avvolgente Lungo, coffee capsules, compatible with nespresso capsule machines, 100 coffee ca
    Nescafe Dolce Gusto slatted Macchiato Caramel 16St capsules
    Tassimo JACOBS coffee shop au lait classico pack of 4
    cafe royal Espresso, coffee, roasted coffee, coffee capsules, nespresso compatible, 132 capsules
    NESCAFE Dolce Gusto slatted Macchiato unsweetened, coffee capsules, without added sugar, Espresso, 3
    Kimbo test package coffee capsules, compatible with Dolce Gusto Nescafe, 4 packs with 16 coffee caps
    Kimbo Cappuccino Napoli coffee capsules, compatible with Dolce Gusto Nescafe, 1 packages with 16 cof
    JSD 400 capsules Caffe bourbon Espresso Point mixture blue compatible with coffee capsules Espresso 
    Nespresso Compatible, 50 x Douwe Egberts Espresso Krach­tig Coffee Capsules, Intensity 10, Sold Loos
    Douwe Egberts Espresso Ristretto, Aluminium Coffee Capsules, Intensity 12, 10 Capsules
    3x Senseo Espresso Intenso 12 Coffee Pod, Holder HD7003 for HD7872
    Tassimo Jacobs Cappuccino, Rainforest Alliance Certified, Pack of 4, 4 x 16 T-Discs (8 Servings)
    Senseo Coffee Pods - Variety Pack 2 x 36 Pods (Total 72 Pods)
    L'Or Espresso Café Forza - Intensité 9 - 50 Capsules en Aluminium Compatibles avec les Machines Nesp
    Tassimo Jacobs Cappuccino, Rainforest Alliance Certified, Pack of 3, 3 x 16 T-Discs (8 Servings)
    nestle japan nescafe dolcegusto special capsule espresso intenso one six(16 the cup)x3 boxed
    coffee capsules small-lunch
    K-Fee Espresto Espresso Decaffeinato, coffee, decaffeinated arabica, intensity 7, 3 boxes, 3x 16 Cap
    Espresso - Ristretto - 20 Capsules aluminum - intensity 11 - coffee
    Espresso - Lungo Profondo - intensity 08 - 20 capsules
    coffee Descaffeinato intensity 3/10 in compatible capsules Dolce taste, coffee ROYAL 16 you.
    Dolce taste 96 capsules chamomile gattopardo
    Dolce taste 96 capsules Orzo gattopardo
    TASSIMO L'OR Espresso Delicious Coffee Capsules Pods t!-Discs 5 Pack, 80 Drinks
    150 capsules coffee Caffe Carbonelli forte - balanced flavor. that 44 mm
    10 capsules coffee from Ginseng nespresso compatible - coffee Kickkick
    """

    overall_pattern = [{'LIKE_NUM': True},
        {'LOWER':{'IN':['grams','g','kg','kilograms','mg','oz','lb','lbs','ml','l','liter','liters' ]}}
        ]
    # unit_pattern = [{'LOWER': {'IN':['grams','g','kg','kilograms','mg','oz','lb','lbs','ml','l','liter','liters' ]}}]
    
    # remove unit_pattern frmo overall pattern to get value of the quantity

    # measure_value_pattern = [{'LIKE_NUM': True},
    #     {'LOWER':{'IN':['grams','g','kg','kilograms','mg','oz','lb','lbs','ml','l','liter','liters' ]}}
    #     ]
    # unit_of_measure_pattern = [{'LOWER':{'IN':['g','kg','grams','kilograms','mg','oz','lb','lbs','ml','l','liter','liters']}}]
    # [{'LIKE_NUM': True, 'DEP': 'nummod'}]



    ## TRAINING: USING INBUILT SPACY REGIMEN

    nlp = spacy.load("en_core_web_sm")

    matcher = Matcher(nlp.vocab)

    matcher.add("Measure", [overall_pattern])

    input_text = [clean_text(input_text) for input_text in input_string.splitlines()] 
    train_data = [parse_train_data(d) for d in nlp.pipe(input_text) if len(matcher(d)) >= 1]
    
    print(train_data[:15])
    
    # Saving the training data to ---> train_data.txt

    file1 = open(r"/home/ananya/Desktop/python/Data_works/productNER/src/train_data_model_2.txt", "w")
    file1.writelines(str(train_data))
    file1.close()

    # @@@@@ Training process @@@@@

    disable_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    losses = {}
    nlp_new= spacy.blank("en")
    nlp_new.add_pipe("ner", last=True)
    ner = nlp_new.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
        #optimizer = nlp.resume_training()
    #optimizer = nlp.create_optimizer()
    optimizer = nlp_new.begin_training()
    for itn in range(20):
            print(f"__________Iteration number{itn}")
            losses = {}
            random.shuffle(train_data)
            # batches = minibatch(train_data, size=compounding(.40,32.0,1.001))
            # for batch in batches:
                #print(batch)
                #for raw_text, entity_offsets in batch:
            for raw_text,entity_offsets in train_data:
                #print(raw_text, entity_offsets)
                doc = nlp.make_doc(raw_text)
                example = Example.from_dict(doc,entity_offsets)
                nlp.update([example], sgd=optimizer, losses=losses, drop=0.3)
            print(f"Losses {losses}")
    nlp.to_disk(r"/home/ananya/Desktop/python/Data_works/OpenCV_OCR/src/NER_experiment/custom_trained_models/with_quantity_measure/model2")

