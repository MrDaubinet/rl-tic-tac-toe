<script>
	// libraries
	import { tick } from 'svelte';
	import { onMount } from 'svelte';
	// Components
	import Board from "../components/Board.svelte"
	import Score from "../components/Score.svelte"
	import Turn from "../components/Turn.svelte"
	import History from "../components/History.svelte"
	import Notification from "../components/Notification.svelte"
	// js class
	import { game } from "../game.js"
	// store
	import history from "../stores/history.ts"

	// Display states
	let showHistory = false
	
	// game states
	let score = game.getScore()
	let agent_play = false
	let outcome = null
	let state
	let turn

	// notification states
	let notification 
	let show_notificaiton = false

	// life Cycle events
	onMount(async () => {
		history.addState({state: [...game.getStates()], turn: game.getTurn()})
		await tick
		updateState()
	});

	// Add to hisory and update board state / turn
	async function updateState() {
		state = $history[$history.length - 1].state
		turn = $history[$history.length - 1].turn
	}

	/* Event triggered by selecting a move */
	async function selectMarker(index) {
		outcome = game.updateState(index)
		score = game.getScore()
		if(outcome == 'win') {
			history.addState({state: [...game.getStates()], turn: game.getTurn()})
			await tick
			updateState()
			showNotification(winnerInfo(game.getWinner()))
			reset()
		} else if(outcome == 'tie') {
			showNotification('Tie')
			reset()
		} else if(outcome == 'invalid') {
			showNotification('Impossible Move')
		} else {
			history.addState({state: [...game.getStates()], turn: game.getTurn()})
			await tick
			updateState()
		}
  }
	
	/* display given text as a notification for 2 seconds */
	function showNotification(new_notification) {
		notification = new_notification
		show_notificaiton = true
		clearTimeout()
		setTimeout(function(){ show_notificaiton = false; }, 2000);
	}

	/* helper function -> format win message */
	function winnerInfo(winner){
		if(winner == 1) 
			return 'Player 1 Wins'
		else {
			if(game.agentPlaying()) 
				return 'Agent Wins'
			else
				return 'Player 2 Wins'
		}
	}

	/* Reset the game state */
	async function reset(timeout=1500){
		clearTimeout()
		setTimeout(async function(){ 
			game.reset()
			history.resetState()
			history.addState({state: [...game.getStates()], turn: game.getTurn()})
			await tick
			updateState()
		}, timeout);
	}
</script>

<div class="h-screen">
	<!-- notification -->
	{#if show_notificaiton}
		<Notification>{notification}</Notification>
	{/if}
	<div 
		class="place-content-center grid mt-20 
			{showHistory ? 'grid-rows-2 grid-cols-none lg:grid-cols-2 lg:grid-rows-none' : 'grid-rows-none grid-cols-none' }
		">
		<div class="px-10 min-w-lg"> 
			<!-- title -->
			<h1 class="pt-10 text-6xl text-center sm:pt-0">Tic-Tac-Toe</h1>
			<div class="py-10">
				<Turn 
					{turn}
					{agent_play}
				/>
			</div>
			<Board 
				state={state}
				{selectMarker}
			/>
			<!-- score -->
			<div class="py-10">
				<Score 
					{score}
					{agent_play}
				/>
			</div>
			<div class="flex justify-center py-5">
				<span class="inline-flex mr-5 rounded-md shadow-sm">
					<button on:click={()=>reset(0)} class="text-4xl cursor-pointer focus:outline-none hover:text-gray-300">
						Reset
					</button>
				</span>
				<span class="inline-flex rounded-md shadow-sm">
					<button on:click={()=>showHistory = !showHistory} class="text-4xl cursor-pointer focus:outline-none hover:text-gray-300">
						{#if showHistory}
							Hide History
						{:else}
							Show History
						{/if}
					</button>
				</span>
			</div>
		</div>
		{#if showHistory}
			<div class="px-10 min-w-lg">
				<h1 class="pt-10 text-6xl text-center sm:pt-0">History</h1>
				<div class="py-10">
					<History 
						{updateState}
					/> 
				</div>
			</div>
		{/if}
	</div>
</div>

