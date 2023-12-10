const express = require('express')
const router = express.Router()
const { ensureAuth, ensureGuest } = require('../middleware/auth')
const place_type = require('../models/PlaceType')

const Entry = require('../models/Entry')
const Nearby = require('../models/NearbySearch')

//@desc Login/Landing page
//@route GET /
router.get('/', ensureGuest, (req, res) => {
    res.render('login', {
        layout: 'login',
    })
})

//@desc Dashboard
//@route GET /dashboard
router.get('/dashboard', ensureAuth, async (req, res) => {
    try {
        const entries = await Entry.find({ user: req.user.id }).lean()
        res.render('dashboard', {
            name: req.user.firstName,
            entries,
        })
    } catch (err) {
        console.error(err)
        res.render('error/500')
    }

})

//@desc     Nearby
//@route    GET /dashboard/nearby
router.get('/dashboard/nearby', ensureAuth, async (req, res) => {
    try {
        const nearby = await Nearby.find({ user: req.user.id }).lean()
        res.render('nearby', {
            name: req.user.firstName,
            nearby,
            placetype: place_type
        })
    } catch (err) {
        console.error(err)
        res.render('error/500')
    }
})

//@desc     Articles
//@route    GET /dashboard/articles
router.get('/dashboard/articles', ensureAuth, async (req, res) => {
    try {
        const articles = [
            {
                title: "Best Places to Live in the U.S. in 2023-2024",
                url: "https://realestate.usnews.com/places/rankings/best-places-to-live",
                author: "U.S. News Real Estate",
                description: "U.S. News analyzed the 150 most populous metro areas to find the best places to live. To make the top of the list, a place had to have good value, be a desirable place to live, have a strong job market and a high quality of life.",
                image: "https://realestate.usnews.com/static-atlas/assets/img/money/real-estate/Badge-Live-Generic.png"
            },
            {
                title: "10 Must-Dos When Renting a Home",
                url: "https://www.moving.com/tips/10-must-dos-when-renting-a-home/",
                author: "Moving.com",
                description: "Renting certainly comes with its fair share of advantages. For one, tenants don’t have to worry about making necessary home improvements to the property. They also aren’t responsible for paying expensive property taxes on the rental. In addition, renters have the option to pick up and move at the end of their lease term, giving them ample flexibility when it comes to moving. However, renters do need to be mindful of quite a few obligations and responsibilities when renting a home. Here are 10 must-dos for all tenants when moving to a rental property.",
                image: "https://i0.wp.com/movingtips.wpengine.com/wp-content/uploads/2018/03/apartment-building.jpg?w=336&ssl=1"
            },
            {
                title: "Rent Guide: A Complete Guide to Renting A Home",
                url: "https://www.zillow.com/learn/category/renting/",
                author: "Zillow",
                description: "Finding an apartment when everyone else seems to be looking for one, too, can test anyone’s optimism. Knowing how to find an apartment for rent quickly and with the least hassle can put you back in the positive zone so you can shift your energy from hunting for a home to moving into one. ",
                image: "https://www.zillowstatic.com/bedrock/app/uploads/sites/47/GettyImages-1330108263_2020-c81f2b-1440x813.jpg"
            },
            {
                title: "Where to Find Used Furniture in Boston (2023)",
                url: "https://massbaymovers.com/blog/used-furniture-in-boston-ma/",
                author: "Mass Bay Movers",
                description: "If you rent a new apartment or buy a townhome in Boston, you may end up needing to find new items to decorate your space. Relocating can already be an expensive process with everything involved, so it’s important to discover ways to save on your moving costs. By getting creative with your decor and furniture, you can enjoy having a beautiful home without paying a high cost.",
                image: "https://massbaymovers.com/wp-content/uploads/2020/04/Boston-e1588453569792.png"
            },
            {
                title: "Top Real Estate Rental Agents",
                url: "https://www.realestaterentalagents.com/",
                author: "Real Estate Rental Agents",
                description: "A comprehensive list of local commercial and residential real estate rental agencies offering homes, houses and apartments for rent. Read unbiased customer reviews and recommendations of local companies.",
                image: "https://www.realestaterentalagents.com/assets/jumbotron/realestaterentalagents.com.png"
            },
            {
                title: "Get Help Paying Rent and Bills",
                url: "https://www.consumerfinance.gov/housing/housing-insecurity/help-for-renters/get-help-paying-rent-and-bills/",
                author: "Consumer Financial Protection Bureau",
                description: "If you’re a renter looking for help with housing costs, you’re not alone. State, tribal, and local organizations distribute money to help renters struggling to keep up with rent and other bills.",
                image: "https://files.consumerfinance.gov/f/images/rent-assistance_670x254.width-800.width-800.jpg"
            }
        ]
        res.render('articles', {
            name: req.user.firstName,
            articles: articles
        })
    } catch (err) {
        console.error(err)
        res.render('error/500')
    }
})

module.exports = router